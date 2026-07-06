# Adding Skeletal Support to DMDX Format

## Current State Analysis

### DMDX Format (src/common/header/files.h, lines 620-655)
The `dmdx_t` structure is an extended MD5 format designed for modern model rendering with multiple meshes and images embedded.

**Current structure:**
```c
typedef struct {
    int ident;
    int version;
    int skinwidth;
    int skinheight;
    int framesize;
    int num_skins;
    int num_xyz;
    int num_st;
    int num_tris;
    int num_glcmds;
    int num_frames;
    int num_meshes;
    int num_imgbit;
    int num_animgroup;
    
    int ofs_skins;
    int ofs_st;
    int ofs_tris;
    int ofs_frames;
    int ofs_glcmds;
    int ofs_meshes;
    int ofs_imgbit;
    int ofs_animgroup;
    int ofs_end;
} dmdx_t;
```

**What it's missing:** No skeletal/bone data structure or frame bone information.

---

## Reference Implementations

### 1. FM-Model Format (fm-model/qdata/qd_Skeletons.h)
**Skeletal Structure:**
- `QD_SkeletalJoint_t`: Contains placement (origin, direction, up in double precision) and rotation
- `Skeletalfmheader_t`: Manages clusters of vertices, with up to 8 clusters
- Per-frame skeletal joints stored in `fmframe_t`
- Support for reference points (attachments)

**Key features:**
- Double-precision placement for accuracy
- Clustered vertex weighting (8 clusters)
- Reference point system for weapon/attachment placement
- Hierarchical organization

### 2. MDR Format (src/common/header/files.h, lines 1310-1406)
**Used by:** EliteForce, Jedi Knight 2, Soldiers of Fortune 2

**Skeletal Structure:**
```c
typedef struct {
    float matrix[3][4];  /* Transformation matrix */
} mdr_bone_t;

typedef struct {
    vec3_t bounds[2];
    vec3_t origin;
    float radius;
    char name[16];
    mdr_bone_t bones[1];  /* Variable-sized array [num_bones] */
} mdr_frame_t;

/* Header fields */
int num_bones;
int ofs_frames;  /* Points to mdr_frame_t array */
```

**Key features:**
- Matrix-based (3x4 affine transformation)
- Simple structure: one bone array per frame
- Works well with vertex weights

### 3. MD5 Format (src/common/models/models_md5.c)
**Internal structures:**
```c
typedef struct {
    char name[64];
    vec3_t pos;
    quat_t orient;
    int parent;
} md5_joint_t;
```

**Key features:**
- Quaternion-based rotation
- Hierarchical (parent pointers)
- Flexible blend controller system

---

## Analysis: Recommended Approach for DMDX

### Option 1: Minimal Addition (Recommended)
**Add basic bone support similar to MDR format:**

1. **Add to `dmdx_t` header:**
   ```c
   int num_bones;        /* Number of bones per frame */
   int ofs_bones;        /* Offset to bone data */
   ```

2. **New structure for bone data per-frame:**
   ```c
   typedef struct {
       float matrix[3][4];  /* Transformation matrix */
       char name[32];       /* Bone name */
       int parent;          /* Parent bone index (-1 = root) */
   } dmdx_bone_t;
   ```

3. **Extend frame structure:**
   - Store `dmdx_bone_t bones[1]` array in each frame
   - Maintain compatibility with existing vertex structure
   - Use for vertex weight transformations

**Advantages:**
- Minimal changes to existing format
- Direct compatibility with MDR approach
- Easy to implement in loader
- Follows existing pattern (mesh_t, animgroup_t)

**File impacts:**
- `src/common/header/files.h`: Add dmdx_bone_t struct and header fields
- `src/common/models/models.c`: Parse bone section
- `src/common/models/models_mdr.c`: Reference for implementation

---

### Option 2: Full FM-Model Integration
**Adopt FM-model's sophisticated skeleton system:**

1. Include cluster-based vertex weighting (8 clusters)
2. Add reference point system
3. Use double-precision placement
4. Support skeletal frame variations

**Advantages:**
- More flexible vertex deformation
- Better animation quality
- Built-in attachment points

**Disadvantages:**
- Significant changes to dmdx_t
- More complex parsing logic
- Larger file sizes

---

### Option 3: Hybrid Approach
**Combine MDR simplicity with FM-model features:**

1. Use MDR's matrix-based bone structure
2. Add FM-model's reference point system
3. Keep vertex weight format simple

---

## Proposed Implementation Steps

### Phase 1: Structure Definition
1. Define `dmdx_bone_t` in `src/common/header/files.h`
2. Add `num_bones` and `ofs_bones` to `dmdx_t`
3. Update version number if needed

### Phase 2: File Format
1. Extend frame data to include bone array
2. Define binary layout for bone data
3. Document offset calculations

### Phase 3: Loader Implementation
1. Modify `src/common/models/models_mdr.c` or create new loader
2. Parse bone data from binary
3. Handle endianness conversions
4. Allocate and populate bone structures

### Phase 4: Vertex Skinning
1. Implement bone-to-vertex weight mapping
2. Integrate with existing vertex transformation
3. Ensure frame preparation handles bones

### Phase 5: Testing
1. Create test models with skeletal data
2. Validate bone hierarchy
3. Check vertex deformation accuracy

---

## Technical Details

### Vertex Weight Structure (Already Exists in MDR)
```c
typedef struct {
    int bone_index;
    float bone_weight;
    vec3_t offset;
} mdr_weight_t;
```

This can be reused with DMDX if added.

### Frame Size Calculations
Current framesize calculation would need to account for bones:
```
framesize = sizeof(daliasxframe_t) + (num_bones * sizeof(dmdx_bone_t))
```

### Offset Management
Sequential offsets needed:
```
ofs_skins → ofs_st → ofs_tris → ofs_frames → ofs_bones → ofs_glcmds → ofs_meshes → ofs_imgbit → ofs_animgroup → ofs_end
```

---

## Compatibility Considerations

### Backward Compatibility
- Existing dmdx files without bones: Set `num_bones = 0`
- Loader must gracefully handle `ofs_bones = 0`
- Version check recommended

### Forward Compatibility
- Leave room for future extensions
- Consider extended bone structures
- Plan for LOD (level of detail) support

---

## Recommendation Summary

**Best approach:** **Option 1 (Minimal Addition)**

1. **Simple & clean:** Follows existing MDR pattern
2. **Maintainable:** Minimal code changes
3. **Performant:** Efficient transformation pipeline
4. **Extensible:** Can add features later (clusters, references)

Add to `dmdx_t`:
```c
int num_bones;        /* 0 if no skeleton */
int ofs_bones;
```

Create `dmdx_bone_t` with matrix-based transformation matching MDR approach.

---

## Files Requiring Changes

1. **src/common/header/files.h**
   - Add dmdx_bone_t structure definition
   - Extend dmdx_t with num_bones and ofs_bones

2. **src/common/models/models.c**
   - Modify Mod_LoadFrames_* to handle bones
   - Update framesize calculations

3. **src/common/models/models_mdr.c** (reference)
   - Reference bone transformation code
   - Adapt MC_UnCompress if using compression

4. **src/common/models/models_utils.c** (new utility functions)
   - Bone data parsing
   - Transformation calculations

---

## References
- FM-Model: `fm-model/qdata/qd_Skeletons.h`
- MDR Format: `src/common/header/files.h` (lines 1310-1406)
- MDR Loader: `src/common/models/models_mdr.c`
- MD5 Format: `src/common/models/models_md5.c`
- DMDX Current: `src/common/header/files.h` (lines 620-655)
