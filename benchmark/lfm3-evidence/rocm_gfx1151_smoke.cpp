#include <hip/hip_runtime.h>
#include <cstdio>
#include <cmath>
__global__ void saxpy(const float *a,const float *b,float *c,int n) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<n)c[i]=2.0f*a[i]+b[i];
}
int main(){
 int count=0;hipError_t e=hipGetDeviceCount(&count);
 if(e!=hipSuccess||count<1){std::fprintf(stderr,"hipGetDeviceCount failed: %s; count=%d\n",hipGetErrorString(e),count);return 2;}
 int n=4096;float *a=nullptr,*b=nullptr,*c=nullptr;
 auto err=[&](hipError_t x,const char* name){if(x!=hipSuccess)std::fprintf(stderr,"%s failed %s\n",name,hipGetErrorString(x));return x==hipSuccess;};
 if(!err(hipMallocManaged(&a,n*sizeof(float)),"alloc a")||!err(hipMallocManaged(&b,n*sizeof(float)),"alloc b")||!err(hipMallocManaged(&c,n*sizeof(float)),"alloc c"))return 3;
 for(int i=0;i<n;i++){a[i]=(float)i;b[i]=3.0f;}
 hipLaunchKernelGGL(saxpy,dim3((n+255)/256),dim3(256),0,0,a,b,c,n);
 if(!err(hipDeviceSynchronize(),"sync"))return 4;
 int bad=0;for(int i=0;i<n;i++)if(std::fabs(c[i]-(2.0f*i+3.0f))>1e-4f)bad++;
 hipDeviceProp_t props{};hipGetDeviceProperties(&props,0);
 std::printf("device=%s, gcn=%s, elements=%d, mismatches=%d\n",props.name,props.gcnArchName,n,bad);
 hipFree(a);hipFree(b);hipFree(c);
 return bad?5:0;
}
