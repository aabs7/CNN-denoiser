?	???eNVS@???eNVS@!???eNVS@	Z???	??Z???	??!Z???	??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$???eNVS@OX?es@A?_[???R@Y??r?????*q=
ף?]@9??v>L?@)      P=2p
9Iterator::Model::Prefetch::BatchV2::ShuffleAndRepeat::Map &??:???!??bk?U@)?? l???1z3?=?T@:Preprocessing2Y
"Iterator::Model::Prefetch::BatchV2O?s????!ZU???X@)$??S ???1?x)
?s!@:Preprocessing2?
OIterator::Model::Prefetch::BatchV2::ShuffleAndRepeat::Map::FlatMap[0]::TFRecord .?l?IF??!?]{??@).?l?IF??1?]{??@:Advanced file read2k
4Iterator::Model::Prefetch::BatchV2::ShuffleAndRepeat ??ip???!@????V@)?L?????1??'??@:Preprocessing2y
BIterator::Model::Prefetch::BatchV2::ShuffleAndRepeat::Map::FlatMap ????????!??6Q?{@)?{)<hv??1??fW_???:Preprocessing2F
Iterator::ModelT1??c??!T???^???)1?Tm7???1il?Q?<??:Preprocessing2P
Iterator::Model::Prefetchw??g??!??}???)w??g??1??}???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 3.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9Z???	??>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	OX?es@OX?es@!OX?es@      ??!       "      ??!       *      ??!       2	?_[???R@?_[???R@!?_[???R@:      ??!       B      ??!       J	??r???????r?????!??r?????R      ??!       Z	??r???????r?????!??r?????JCPU_ONLYYZ???	??b Y      Y@qB?^I@??"?
both?Your program is POTENTIALLY input-bound because 3.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQ2"CPU: B 