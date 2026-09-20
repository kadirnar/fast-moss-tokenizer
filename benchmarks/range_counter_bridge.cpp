// Minimal CUPTI range-profiler bridge. Uses installed NVIDIA headers/libraries.
#include <cuda.h>
#include <cupti_profiler_target.h>
#include <cupti_profiler_host.h>
#include <cupti_range_profiler.h>
#include <cupti_target.h>
#include <vector>
#include <string>
#include <stdexcept>

static std::string error;
static std::string warning;
static CUpti_Profiler_Host_Object* host=nullptr;
static CUpti_RangeProfiler_Object* target=nullptr;
static std::vector<uint8_t> config,data;
static std::vector<const char*> metrics;
#define INIT(T,x) T x={}; x.structSize=T##_STRUCT_SIZE
#define CHECK(call) do {auto rc=(call); if(rc!=CUPTI_SUCCESS){const char* s=nullptr; cuptiGetResultString(rc,&s); throw std::runtime_error(std::string(#call)+": "+(s?s:"unknown"));}} while(0)
extern "C" const char* counter_error(){return error.c_str();}
extern "C" const char* counter_warning(){return warning.c_str();}
extern "C" int counter_close(){
    int result=0;
    if(target){INIT(CUpti_RangeProfiler_Disable_Params,p);p.pRangeProfilerObject=target;result=cuptiRangeProfilerDisable(&p);target=nullptr;}
    if(host){INIT(CUpti_Profiler_Host_Deinitialize_Params,p);p.pHostObject=host;int rc=cuptiProfilerHostDeinitialize(&p);if(!result)result=rc;host=nullptr;}
    config.clear();data.clear();metrics.clear();return result;
}
extern "C" int counter_setup(void* context,const char** names,size_t count){
 try{
    error.clear();warning.clear();metrics.assign(names,names+count);
    INIT(CUpti_Profiler_Initialize_Params,init);CHECK(cuptiProfilerInitialize(&init));
    INIT(CUpti_Device_GetChipName_Params,chip);chip.deviceIndex=0;CHECK(cuptiDeviceGetChipName(&chip));
    INIT(CUpti_Profiler_GetCounterAvailability_Params,available);available.ctx=(CUcontext)context;
    CHECK(cuptiProfilerGetCounterAvailability(&available));
    std::vector<uint8_t> availability(available.counterAvailabilityImageSize);
    available.pCounterAvailabilityImage=availability.data();
    auto availableResult=cuptiProfilerGetCounterAvailability(&available);
    if(availableResult!=CUPTI_SUCCESS){const char* text=nullptr;cuptiGetResultString(availableResult,&text);warning=std::string("Optional availability image: ")+(text?text:"unknown");availability.clear();}
    INIT(CUpti_Profiler_Host_Initialize_Params,h);h.profilerType=CUPTI_PROFILER_TYPE_RANGE_PROFILER;
    h.pChipName=chip.pChipName;h.pCounterAvailabilityImage=availability.empty()?nullptr:availability.data();CHECK(cuptiProfilerHostInitialize(&h));host=h.pHostObject;
    INIT(CUpti_Profiler_Host_ConfigAddMetrics_Params,add);add.pHostObject=host;add.ppMetricNames=names;add.numMetrics=count;CHECK(cuptiProfilerHostConfigAddMetrics(&add));
    INIT(CUpti_Profiler_Host_GetConfigImageSize_Params,size);size.pHostObject=host;CHECK(cuptiProfilerHostGetConfigImageSize(&size));config.resize(size.configImageSize);
    INIT(CUpti_Profiler_Host_GetConfigImage_Params,image);image.pHostObject=host;image.configImageSize=config.size();image.pConfigImage=config.data();CHECK(cuptiProfilerHostGetConfigImage(&image));
    INIT(CUpti_RangeProfiler_Enable_Params,enable);enable.ctx=(CUcontext)context;CHECK(cuptiRangeProfilerEnable(&enable));target=enable.pRangeProfilerObject;
    INIT(CUpti_RangeProfiler_GetCounterDataSize_Params,ds);ds.pRangeProfilerObject=target;ds.pMetricNames=names;ds.numMetrics=count;ds.maxNumOfRanges=1;ds.maxNumRangeTreeNodes=1;CHECK(cuptiRangeProfilerGetCounterDataSize(&ds));data.resize(ds.counterDataSize);
    INIT(CUpti_RangeProfiler_CounterDataImage_Initialize_Params,di);di.pRangeProfilerObject=target;di.counterDataSize=data.size();di.pCounterData=data.data();CHECK(cuptiRangeProfilerCounterDataImageInitialize(&di));
    INIT(CUpti_RangeProfiler_SetConfig_Params,set);set.pRangeProfilerObject=target;set.configSize=config.size();set.pConfig=config.data();set.counterDataImageSize=data.size();set.pCounterDataImage=data.data();
    set.range=CUPTI_UserRange;set.replayMode=CUPTI_UserReplay;set.maxRangesPerPass=1;set.numNestingLevels=1;set.minNestingLevel=1;set.targetNestingLevel=0;
    CHECK(cuptiRangeProfilerSetConfig(&set));return 0;
 }catch(const std::exception& e){error=e.what();counter_close();return -1;}
}
extern "C" int counter_start(){
 try{INIT(CUpti_RangeProfiler_Start_Params,s);s.pRangeProfilerObject=target;CHECK(cuptiRangeProfilerStart(&s));
     INIT(CUpti_RangeProfiler_PushRange_Params,p);p.pRangeProfilerObject=target;p.pRangeName="gemv";CHECK(cuptiRangeProfilerPushRange(&p));return 0;
 }catch(const std::exception& e){error=e.what();return -1;}
}
extern "C" int counter_stop(){
 try{INIT(CUpti_RangeProfiler_PopRange_Params,p);p.pRangeProfilerObject=target;CHECK(cuptiRangeProfilerPopRange(&p));
     INIT(CUpti_RangeProfiler_Stop_Params,s);s.pRangeProfilerObject=target;CHECK(cuptiRangeProfilerStop(&s));return s.isAllPassSubmitted?1:0;
 }catch(const std::exception& e){error=e.what();return -1;}
}
extern "C" int counter_values(double* values){
 try{INIT(CUpti_RangeProfiler_DecodeData_Params,d);d.pRangeProfilerObject=target;CHECK(cuptiRangeProfilerDecodeData(&d));
     if(d.numOfRangeDropped)throw std::runtime_error("CUPTI dropped ranges");
     INIT(CUpti_RangeProfiler_GetCounterDataInfo_Params,i);i.pCounterDataImage=data.data();i.counterDataImageSize=data.size();CHECK(cuptiRangeProfilerGetCounterDataInfo(&i));
     if(i.numTotalRanges!=1)throw std::runtime_error("Expected exactly one collected range");
     INIT(CUpti_Profiler_Host_EvaluateToGpuValues_Params,e);e.pHostObject=host;e.pCounterDataImage=data.data();e.counterDataImageSize=data.size();e.rangeIndex=0;e.ppMetricNames=metrics.data();e.numMetrics=metrics.size();e.pMetricValues=values;CHECK(cuptiProfilerHostEvaluateToGpuValues(&e));return 0;
 }catch(const std::exception& e){error=e.what();return -1;}
}
