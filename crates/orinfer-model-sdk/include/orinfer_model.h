/* orinfer model ABI v1. All spans are borrowed until the corresponding free
 * callback; errors and output allocations must be freed by their producer. */
#ifndef ORINFER_MODEL_H
#define ORINFER_MODEL_H
#include <stddef.h>
#include <stdint.h>
#define ORINFER_MODEL_ABI 1
#define ORINFER_RUNTIME_ABI 1
typedef struct { const uint8_t *data; size_t len; } OrinferBytes;
typedef struct { uint8_t *data; size_t len; } OrinferOwnedBytes;
typedef struct { size_t slot, tokens; } OrinferSegment;
typedef struct { OrinferBytes name, buffer; size_t offset; } OrinferView;
typedef struct { size_t index; int32_t value; } OrinferArgument;
typedef struct {
    uint32_t kind;
    OrinferBytes name, destination;
    size_t bytes, sequence;
    uint32_t has_launch, grid[3];
    const OrinferArgument *arguments;
    size_t argument_count;
    const OrinferView *views;
    size_t view_count;
} OrinferCommand;
typedef struct { uint32_t present; size_t sequence; OrinferBytes buffer; } OrinferBinding;
typedef struct {
    void *owner;
    const OrinferCommand *commands;
    size_t command_count;
    const OrinferBinding *bindings;
    size_t binding_count;
} OrinferBatch;
typedef struct { size_t grid_height, grid_width; } OrinferImageGrid;
typedef struct {
    void *owner;
    const int32_t *indices;
    size_t index_count;
    const uint32_t *positions;
    size_t position_count;
} OrinferVisual;
typedef struct {
    uint32_t abi_version;
    size_t struct_size;
    int32_t (*describe)(OrinferOwnedBytes *);
    int32_t (*create)(const uint8_t *, size_t, void **, OrinferOwnedBytes *);
    void (*destroy)(void *);
    int32_t (*batch)(void *, const OrinferSegment *, size_t, uint32_t, OrinferBatch *, OrinferOwnedBytes *);
    void (*free_bytes)(OrinferOwnedBytes);
    void (*free_batch)(OrinferBatch);
    int32_t (*visual)(void *, const uint32_t *, size_t, const OrinferImageGrid *, size_t, size_t, OrinferVisual *, OrinferOwnedBytes *);
    void (*free_visual)(OrinferVisual);
} OrinferModelApi;
const OrinferModelApi *orinfer_model_v1(uint32_t version);
/* Optional independent extension; preserves the original model ABI table. */
typedef struct { OrinferBytes buffer, data; } OrinferInputUpload;
typedef struct { void *owner; const OrinferInputUpload *uploads; size_t count; } OrinferInputs;
typedef struct {
    uint32_t version;
    size_t struct_size;
    int32_t (*prepare)(void *, const uint32_t *, size_t, const uint32_t *, size_t, OrinferInputs *, OrinferOwnedBytes *);
    void (*free)(OrinferInputs);
    int32_t (*prepare_program)(void *, OrinferBytes, const uint32_t *, size_t, const uint32_t *, size_t, OrinferInputs *, OrinferOwnedBytes *);
} OrinferInputApi;
const OrinferInputApi *orinfer_model_inputs_v1(uint32_t version);
/* Native execution export. Parameter kinds: buffer, slice, i32, u32, i64,
 * u64, f32 = 0..6. Signed integers/floats use their exact unsigned bit pattern.
 * Definition/dynamic indices refer to the validated catalog; SIZE_MAX means
 * no dynamic specialization. All strings and nested arrays live until free. */
typedef struct { uint32_t kind; OrinferBytes name; size_t offset; uint64_t scalar; } OrinferPlanArgument;
typedef struct {
    OrinferBytes name;
    size_t definition;
    uint32_t grid[3];
    const OrinferPlanArgument *arguments;
    size_t argument_count, dynamic;
    uint32_t plan_contract;
} OrinferPlanKernel;
typedef struct { uint32_t kind; OrinferBytes name, destination; size_t bytes; } OrinferPlanCommand;
typedef struct { OrinferBytes name; const OrinferPlanCommand *commands; size_t command_count; } OrinferPlanProgram;
typedef struct {
    void *owner;
    const OrinferPlanKernel *kernels;
    size_t kernel_count;
    const OrinferPlanProgram *programs;
    size_t program_count;
} OrinferPlans;
typedef struct {
    uint32_t version;
    size_t struct_size;
    size_t kernel_size, argument_size, program_size, command_size;
    int32_t (*export_plans)(void *, OrinferPlans *, OrinferOwnedBytes *);
    void (*free)(OrinferPlans);
} OrinferPlanApi;
const OrinferPlanApi *orinfer_model_plans_v1(uint32_t version);
#endif
