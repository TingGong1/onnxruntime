import onnx
from onnx_model import OnnxModel
#model_path = r'd:\CompliantModels\NLG_optimization\nlg_utils\tmp_result\distilgput-17B-split-2_logitspushing\part_24_48\model.onnx'
model_path = r'D:\CompliantModels\DeepWrite_all\converted_onnx_batchsize_incompletemask_tianlei_optimized\multistreamDW_chk_GPT2LMHeadModel_ConfigurableOneStepSearch_past_fp16\multistreamDW_chk_GPT2LMHeadModel_ConfigurableOneStepSearch_past_fp16.onnx'
# model_path = 'query_rewrite/huggingface_past_fp32/huggingface_past_fp32.onnx'
model = onnx.load(model_path)
o_model = OnnxModel(model)
use_external_data_format=True
ai_onnx_domain = [
    opset for opset in o_model.model.opset_import if not opset.domain or opset.domain == "ai.onnx"
]
o_model.model.opset_import.remove(ai_onnx_domain[0])
o_model.model.opset_import.extend([onnx.helper.make_opsetid("", 14)])
#opt_path = r'd:\CompliantModels\NLG_optimization\nlg_utils\tmp_result\distilgput-17B-split-2_logitspushing\part_24_48\model_opset14.onnx'
opt_path = r'D:\CompliantModels\DeepWrite_all\converted_onnx_batchsize_incompletemask_tianlei_optimized\multistreamDW_chk_GPT2LMHeadModel_ConfigurableOneStepSearch_past_fp16_op14\multistreamDW_chk_GPT2LMHeadModel_ConfigurableOneStepSearch_past_fp16_op14.onnx'
o_model.save_model_to_file(opt_path, use_external_data_format=use_external_data_format)
