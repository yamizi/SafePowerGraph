import os
from hta.trace_analysis import TraceAnalysis

trace_dir = "output/trace/"
os.makedirs(trace_dir,exist_ok=True)
analyzer = TraceAnalysis(trace_dir= trace_dir)

# Memory bandwidth summary
memory_bw_summary = analyzer.get_memory_bw_summary()

temporal_breakdown_df = analyzer.get_temporal_breakdown()