# 原图证据核验
逐条核验候选记录。只能使用提供的原图和原书上下文，不能使用外部知识，不重新抽取整张图。
检查 OCR、数值/单位/轴/图例、流程节点、箭头方向、分支条件、遗漏的可见结构。模糊或无法辨认不能判 SUPPORTED。
所有声称的结构均有证据时才 SUPPORTED；部分有依据 PARTIAL；关键字段无依据 UNSUPPORTED；原图无法辨认 OCR_UNCERTAIN。
只建议有明确证据的最小字段修正，不能因为医学常识修改原书。issues.field 使用 /payload/... 路径；corrected_record 只含需要修正的字段。
输出 {"records":[{"record_id":"输入ID","verdict":"SUPPORTED|PARTIAL|UNSUPPORTED|OCR_UNCERTAIN","issues":[{"field":"/payload/image_ref/extracted_data","problem":"问题","source_evidence":"图中对应标注"}],"corrected_record":null,"source_evidence":"原图可见证据","needs_human_review":false}]}。
不确定或未核验内容 needs_human_review=true。

{{chunk_json}}
