"""Profiled dense BTC transposes, sharing the validated native Welford arithmetic.

Only the input addressing and optional cooperative shared-memory staging change.
The resulting output and statistics keep the contiguous native layout.
"""
from .normalization import SOURCE as BASE

CONFIGS = {
    (1, 2, 768): (False, 128, 1, False, 0),
    (1, 3, 1280): (False, 128, 1, False, 0),
    (1, 4, 768): (False, 128, 1, False, 0),
    (1, 6, 768): (False, 128, 1, False, 0),
    (1, 12, 768): (False, 128, 1, False, 0),
    (1, 40, 1280): (False, 128, 1, False, 0),
    (1, 80, 768): (True, 64, 4, False, 0),
    (1, 160, 768): (True, 128, 1, True, 0),
    (2, 3, 1280): (False, 128, 1, False, 0),
    (2, 6, 768): (False, 128, 1, False, 0),
    (2, 12, 768): (False, 128, 1, False, 0),
    (8, 2, 768): (False, 128, 1, False, 0),
    (8, 3, 1280): (False, 128, 1, False, 0),
    (8, 4, 768): (False, 128, 1, False, 0),
    (8, 6, 768): (False, 128, 1, False, 0),
    (8, 12, 768): (True, 128, 4, False, 0),
    (128, 2, 768): (False, 128, 1, False, 0),
    (128, 4, 768): (True, 128, 4, True, 0),
}

# Fail explicitly if a future arithmetic edit changes any addressing splice.
for pattern in (
    'float4 v=reinterpret_cast<const float4*>(x)[logical+step*128];',
    'float4 v=reinterpret_cast<const float4*>(X+row*N)[idx];',
    '    if(row>=rows)return;',
):
    if BASE.count(pattern) != 1:
        raise RuntimeError('Strided LayerNorm requires the validated input-addressing sites')
if BASE.count('X+row*N,logical') != 2:
    raise RuntimeError('Strided LayerNorm requires both validated statistics addressing sites')

SOURCE=BASE.replace('float4 v=reinterpret_cast<const float4*>(x)[logical+step*128];',
                    'float4 v=read4(x,logical+step*128);')
SOURCE=SOURCE.replace('X+row*N,logical','row_ptr,logical')
SOURCE=SOURCE.replace('float4 v=reinterpret_cast<const float4*>(X+row*N)[idx];','float4 v=read4(row_ptr,idx);')
SOURCE=SOURCE.replace('    if(row>=rows)return;',r'''
    #if STAGE
    constexpr int R=THREADS/32;
    __shared__ float tile[R*(N+PAD)];
    for(int index=threadIdx.x;index<N*R;index+=THREADS){
        int c=index/R,r=index%R,global_row=blockIdx.x*R+r;
        tile[r*(N+PAD)+c]=global_row<rows ? X[(global_row/T)*N*T+(global_row%T)+c*T] : 0.f;
    }
    __syncthreads();
    const float* row_ptr=tile+warp*(N+PAD);
    #else
    const float* row_ptr=X+(row/T)*N*T+(row%T);
    #endif
    if(row>=rows)return;''')
SOURCE=r'''
__device__ __forceinline__ float4 read4(const float* x,int vector){
    #if STAGE
    return reinterpret_cast<const float4*>(x)[vector];
    #else
    int c=vector*4;
    return {x[c*T],x[(c+1)*T],x[(c+2)*T],x[(c+3)*T]};
    #endif
}
'''+SOURCE
