# 原书图片结构化抽取
忠实分析提供的原图及图注，不使用外部知识，不猜测模糊小字，不根据图注虚构图中数据。
每张图输出 FIGURE 记录；图中多个独立事实放入 extracted_data.facts，保留各自条件和可用的证据区域。
流程图保留 steps、decision_points、relations、entry_point、terminal_points，分支条件和箭头方向必须逐项保留；图表保留标题、轴、单位、图例、data_points；表格截图保留 headers/rows/notes。
只对能辨认的内容结构化；不确定内容写 ambiguities，并降低 confidence。不推算未标注数值。
输出 {"records":[{"record_type":"FIGURE","raw_text":"图中支持记录的实际文字","image_ref":{"asset_path":"","figure_type":"流程图/图表/表格截图/其他","description":"忠实描述","extracted_data":{},"confidence":0.8,"ocr_text":"可辨认文字"},"attributes":{},"ambiguities":[]}]}。
如果坐标无法可靠确定则不输出坐标。图注与图像冲突时保留冲突，不自行修正。

{{chunk_json}}
