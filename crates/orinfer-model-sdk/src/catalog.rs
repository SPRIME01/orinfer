//! Compiled implementations are independent of tensor roles and model layers.
use crate::{
    artifact::{Argument, FileIdentity, Kernel, Result},
    operators::dynamic::{DynamicBatchKernel, RowArgument, RowExpression},
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ParameterKind {
    Buffer,
    BufferSlice,
    I32,
    U32,
    I64,
    U64,
    F32,
}
impl ParameterKind {
    pub fn of(argument: &Argument) -> Self {
        match argument {
            Argument::Buffer { .. } => Self::Buffer,
            Argument::BufferSlice { .. } => Self::BufferSlice,
            Argument::I32 { .. } => Self::I32,
            Argument::U32 { .. } => Self::U32,
            Argument::I64 { .. } => Self::I64,
            Argument::U64 { .. } => Self::U64,
            Argument::F32 { .. } => Self::F32,
        }
    }
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Definition {
    pub module: usize,
    pub source: usize,
    pub host_abi: usize,
    pub symbol: String,
    pub block: [u32; 3],
    pub shared_memory_bytes: u32,
    pub cooperative: bool,
    pub parameters: Vec<ParameterKind>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub readonly_arguments: Vec<bool>,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DynamicDefinition {
    pub capacity: usize,
    pub grid: [RowExpression; 3],
    pub arguments: Vec<RowArgument>,
}
impl DynamicDefinition {
    pub fn bind(&self, name: &str) -> DynamicBatchKernel {
        DynamicBatchKernel {
            name: name.into(),
            capacity: self.capacity,
            grid: self.grid.clone(),
            arguments: self.arguments.clone(),
        }
    }
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Catalog {
    pub assets: Vec<FileIdentity>,
    pub definitions: Vec<Definition>,
    pub dynamic: Vec<DynamicDefinition>,
    /// Seals the generated native role/implementation mapping, not weight contents.
    pub binding_signature: String,
}
impl Catalog {
    pub fn validate(&self) -> Result<()> {
        if self.binding_signature.len() != 64
            || !self
                .binding_signature
                .bytes()
                .all(|v| v.is_ascii_hexdigit() && !v.is_ascii_uppercase())
            || self.definitions.is_empty()
        {
            return Err("Invalid native binding signature or empty kernel catalog".into());
        }
        for d in &self.definitions {
            if [d.module, d.source, d.host_abi]
                .iter()
                .any(|&i| i >= self.assets.len())
                || d.symbol.is_empty()
                || d.block.contains(&0)
                || d.parameters.len() > 256
                || (!d.readonly_arguments.is_empty()
                    && d.readonly_arguments.len() != d.parameters.len())
            {
                return Err("Invalid compiled kernel definition".into());
            }
        }
        Ok(())
    }
    pub fn bind(
        &self,
        definition: usize,
        name: String,
        grid: [u32; 3],
        args: Vec<Argument>,
    ) -> Result<Kernel> {
        let d = self
            .definitions
            .get(definition)
            .ok_or("Unknown compiled kernel definition")?;
        if grid.contains(&0)
            || d.parameters.len() != args.len()
            || d.parameters
                .iter()
                .zip(&args)
                .any(|(&kind, a)| kind != ParameterKind::of(a))
        {
            return Err(format!(
                "{name}: native binding differs from compiled parameter ABI"
            ));
        }
        let asset = |i| {
            self.assets
                .get(i)
                .cloned()
                .ok_or_else(|| "Unknown kernel asset".to_owned())
        };
        Ok(Kernel {
            name,
            grid,
            args,
            module: asset(d.module)?,
            source: asset(d.source)?,
            host_abi: asset(d.host_abi)?,
            symbol: d.symbol.clone(),
            block: d.block,
            shared_memory_bytes: d.shared_memory_bytes,
            cooperative: d.cooperative,
            readonly_arguments: d.readonly_arguments.clone(),
        })
    }
}

/// Build-time generated role references. Offsets remain explicit and checked by
/// the common runtime against the model's physical tensor layout.
pub enum RoleArgument {
    Buffer(usize),
    BufferSlice(usize, usize),
    I32(i32),
    U32(u32),
    I64(i64),
    U64(u64),
    F32(u32),
}
impl RoleArgument {
    pub fn bind(&self, roles: &[&str]) -> Result<Argument> {
        let role = |i: usize| {
            roles
                .get(i)
                .map(|s| (*s).to_owned())
                .ok_or_else(|| "Unknown native tensor role".to_owned())
        };
        Ok(match *self {
            Self::Buffer(i) => Argument::Buffer { name: role(i)? },
            Self::BufferSlice(i, offset) => Argument::BufferSlice {
                name: role(i)?,
                offset,
            },
            Self::I32(value) => Argument::I32 { value },
            Self::U32(value) => Argument::U32 { value },
            Self::I64(value) => Argument::I64 { value },
            Self::U64(value) => Argument::U64 { value },
            Self::F32(bits) => Argument::F32 {
                value: f32::from_bits(bits),
            },
        })
    }
}
pub struct RoleBinding {
    pub name: &'static str,
    pub definition: usize,
    pub grid: [u32; 3],
    pub arguments: usize,
    pub dynamic: Option<usize>,
    pub plan_contract: bool,
}
