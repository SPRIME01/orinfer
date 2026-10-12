//! Bulk native bindings and programs. No Rust type crosses the library boundary.
use super::{Bytes, ModelImplementation, OwnedBytes, boundary};
use crate::{artifact::Argument as Value, model::Operation};
use std::{collections::BTreeMap, ffi::c_void, ptr};
pub const ENTRYPOINT: &[u8] = b"orinfer_model_plans_v1\0";
pub struct KernelBinding {
    pub name: String,
    pub definition: usize,
    pub grid: [u32; 3],
    pub args: Vec<Value>,
    pub dynamic: Option<usize>,
    pub plan_contract: bool,
}
pub struct Execution {
    pub kernels: Vec<KernelBinding>,
    pub programs: BTreeMap<String, Vec<Operation>>,
}
#[repr(C)]
pub struct Argument {
    pub kind: u32,
    pub name: Bytes,
    pub offset: usize,
    pub scalar: u64,
}
#[repr(C)]
pub struct Kernel {
    pub name: Bytes,
    pub definition: usize,
    pub grid: [u32; 3],
    pub arguments: *const Argument,
    pub argument_count: usize,
    pub dynamic: usize,
    pub plan_contract: u32,
}
#[repr(C)]
pub struct Command {
    pub kind: u32,
    pub name: Bytes,
    pub destination: Bytes,
    pub bytes: usize,
}
#[repr(C)]
pub struct Program {
    pub name: Bytes,
    pub commands: *const Command,
    pub command_count: usize,
}
#[repr(C)]
pub struct Output {
    pub owner: *mut c_void,
    pub kernels: *const Kernel,
    pub kernel_count: usize,
    pub programs: *const Program,
    pub program_count: usize,
}
impl Default for Output {
    fn default() -> Self {
        Self {
            owner: ptr::null_mut(),
            kernels: ptr::null(),
            kernel_count: 0,
            programs: ptr::null(),
            program_count: 0,
        }
    }
}
#[repr(C)]
pub struct Api {
    pub version: u32,
    pub struct_size: usize,
    pub kernel_size: usize,
    pub argument_size: usize,
    pub program_size: usize,
    pub command_size: usize,
    pub export: unsafe extern "C" fn(*mut c_void, *mut Output, *mut OwnedBytes) -> i32,
    pub free: unsafe extern "C" fn(Output),
}
struct Storage {
    execution: Execution,
    arguments: Vec<Vec<Argument>>,
    kernels: Vec<Kernel>,
    commands: Vec<Vec<Command>>,
    programs: Vec<Program>,
}
impl Storage {
    fn export(execution: Execution) -> Output {
        let mut owner = Box::new(Self {
            execution,
            arguments: vec![],
            kernels: vec![],
            commands: vec![],
            programs: vec![],
        });
        let empty = || Bytes::borrowed("");
        for k in &owner.execution.kernels {
            let args: Vec<_> = k
                .args
                .iter()
                .map(|a| {
                    let mut raw = Argument {
                        kind: 0,
                        name: empty(),
                        offset: 0,
                        scalar: 0,
                    };
                    match a {
                        Value::Buffer { name } => {
                            raw.name = Bytes::borrowed(name);
                        }
                        Value::BufferSlice { name, offset } => {
                            raw.kind = 1;
                            raw.name = Bytes::borrowed(name);
                            raw.offset = *offset;
                        }
                        Value::I32 { value } => {
                            raw.kind = 2;
                            raw.scalar = *value as u32 as u64;
                        }
                        Value::U32 { value } => {
                            raw.kind = 3;
                            raw.scalar = *value as u64;
                        }
                        Value::I64 { value } => {
                            raw.kind = 4;
                            raw.scalar = *value as u64;
                        }
                        Value::U64 { value } => {
                            raw.kind = 5;
                            raw.scalar = *value;
                        }
                        Value::F32 { value } => {
                            raw.kind = 6;
                            raw.scalar = value.to_bits() as u64;
                        }
                    }
                    raw
                })
                .collect();
            owner.kernels.push(Kernel {
                name: Bytes::borrowed(&k.name),
                definition: k.definition,
                grid: k.grid,
                arguments: args.as_ptr(),
                argument_count: args.len(),
                dynamic: k.dynamic.unwrap_or(usize::MAX),
                plan_contract: u32::from(k.plan_contract),
            });
            owner.arguments.push(args);
        }
        for (name, operations) in &owner.execution.programs {
            let commands: Vec<_> = operations
                .iter()
                .map(|op| match op {
                    Operation::Kernel { name } => Command {
                        kind: 0,
                        name: Bytes::borrowed(name),
                        destination: empty(),
                        bytes: 0,
                    },
                    Operation::Copy {
                        source,
                        destination,
                        bytes,
                    } => Command {
                        kind: 1,
                        name: Bytes::borrowed(source),
                        destination: Bytes::borrowed(destination),
                        bytes: *bytes,
                    },
                    Operation::Zero { destination, bytes } => Command {
                        kind: 2,
                        name: empty(),
                        destination: Bytes::borrowed(destination),
                        bytes: *bytes,
                    },
                })
                .collect();
            owner.programs.push(Program {
                name: Bytes::borrowed(name),
                commands: commands.as_ptr(),
                command_count: commands.len(),
            });
            owner.commands.push(commands);
        }
        Output {
            kernels: owner.kernels.as_ptr(),
            kernel_count: owner.kernels.len(),
            programs: owner.programs.as_ptr(),
            program_count: owner.programs.len(),
            owner: Box::into_raw(owner).cast(),
        }
    }
}
/// # Safety
/// The handle belongs to M; output/error are writable for this synchronous call.
pub unsafe extern "C" fn export<M: ModelImplementation>(
    handle: *mut c_void,
    output: *mut Output,
    error: *mut OwnedBytes,
) -> i32 {
    boundary(error, || {
        if handle.is_null() || output.is_null() {
            return Err("Invalid native plan export arguments".into());
        }
        // SAFETY: caller retains this library's model handle until destroy.
        let execution = unsafe { &*handle.cast::<M>() }.execution()?;
        // SAFETY: output is writable; free returns exactly this owner to its library.
        unsafe {
            ptr::write(output, Storage::export(execution));
        }
        Ok(vec![])
    })
}
/// # Safety
/// output must be returned by export in this library and released exactly once.
pub unsafe extern "C" fn free(output: Output) {
    if !output.owner.is_null() {
        // SAFETY: private Storage is created and destroyed in the same library.
        drop(unsafe { Box::from_raw(output.owner.cast::<Storage>()) });
    }
}
