Search.setIndex({docnames:["api_en/index_en","api_en/modules","api_en/paddleslim","api_en/paddleslim.analysis","api_en/paddleslim.common","api_en/paddleslim.core","api_en/paddleslim.dist","api_en/paddleslim.models","api_en/paddleslim.nas","api_en/paddleslim.nas.one_shot","api_en/paddleslim.nas.search_space","api_en/paddleslim.pantheon","api_en/paddleslim.prune","api_en/paddleslim.quant","index","index_en","install_en","intro_en","model_zoo_en","quick_start/index_en","quick_start/pruning_tutorial_en","quick_start/quant_aware_tutorial_en","quick_start/quant_post_tutorial_en","tutorials/index_en","tutorials/sensitivity_tutorial_en"],envversion:50,filenames:["api_en/index_en.rst","api_en/modules.rst","api_en/paddleslim.rst","api_en/paddleslim.analysis.rst","api_en/paddleslim.common.rst","api_en/paddleslim.core.rst","api_en/paddleslim.dist.rst","api_en/paddleslim.models.rst","api_en/paddleslim.nas.rst","api_en/paddleslim.nas.one_shot.rst","api_en/paddleslim.nas.search_space.rst","api_en/paddleslim.pantheon.rst","api_en/paddleslim.prune.rst","api_en/paddleslim.quant.rst","index.rst","index_en.rst","install_en.md","intro_en.md","model_zoo_en.md","quick_start/index_en.rst","quick_start/pruning_tutorial_en.md","quick_start/quant_aware_tutorial_en.md","quick_start/quant_post_tutorial_en.md","tutorials/index_en.rst","tutorials/sensitivity_tutorial_en.md"],objects:{"":{paddleslim:[2,0,0,"-"]},"paddleslim.analysis":{LatencyEvaluator:[3,1,1,""],TableLatencyEvaluator:[3,1,1,""],flops:[3,0,0,"-"],latency:[3,0,0,"-"],model_size:[3,0,0,"-"]},"paddleslim.analysis.LatencyEvaluator":{latency:[3,2,1,""]},"paddleslim.analysis.TableLatencyEvaluator":{latency:[3,2,1,""]},"paddleslim.analysis.flops":{flops:[3,3,1,""]},"paddleslim.analysis.latency":{LatencyEvaluator:[3,1,1,""],TableLatencyEvaluator:[3,1,1,""]},"paddleslim.analysis.latency.LatencyEvaluator":{latency:[3,2,1,""]},"paddleslim.analysis.latency.TableLatencyEvaluator":{latency:[3,2,1,""]},"paddleslim.analysis.model_size":{model_size:[3,3,1,""]},"paddleslim.common":{ControllerClient:[4,1,1,""],ControllerServer:[4,1,1,""],EvolutionaryController:[4,1,1,""],SAController:[4,1,1,""],cached_reader:[4,0,0,"-"],controller:[4,0,0,"-"],controller_client:[4,0,0,"-"],controller_server:[4,0,0,"-"],get_logger:[4,3,1,""],lock:[4,0,0,"-"],log_helper:[4,0,0,"-"],sa_controller:[4,0,0,"-"],unlock:[4,3,1,""]},"paddleslim.common.ControllerClient":{next_tokens:[4,2,1,""],request_current_info:[4,2,1,""],update:[4,2,1,""]},"paddleslim.common.ControllerServer":{close:[4,2,1,""],ip:[4,2,1,""],port:[4,2,1,""],run:[4,2,1,""],start:[4,2,1,""]},"paddleslim.common.EvolutionaryController":{next_tokens:[4,2,1,""],reset:[4,2,1,""],update:[4,2,1,""]},"paddleslim.common.SAController":{best_tokens:[4,4,1,""],current_tokens:[4,4,1,""],max_reward:[4,4,1,""],next_tokens:[4,2,1,""],update:[4,2,1,""]},"paddleslim.common.cached_reader":{cached_reader:[4,3,1,""]},"paddleslim.common.controller":{EvolutionaryController:[4,1,1,""]},"paddleslim.common.controller.EvolutionaryController":{next_tokens:[4,2,1,""],reset:[4,2,1,""],update:[4,2,1,""]},"paddleslim.common.controller_client":{ControllerClient:[4,1,1,""]},"paddleslim.common.controller_client.ControllerClient":{next_tokens:[4,2,1,""],request_current_info:[4,2,1,""],update:[4,2,1,""]},"paddleslim.common.controller_server":{ControllerServer:[4,1,1,""]},"paddleslim.common.controller_server.ControllerServer":{close:[4,2,1,""],ip:[4,2,1,""],port:[4,2,1,""],run:[4,2,1,""],start:[4,2,1,""]},"paddleslim.common.lock":{lock:[4,3,1,""],unlock:[4,3,1,""]},"paddleslim.common.log_helper":{get_logger:[4,3,1,""]},"paddleslim.common.sa_controller":{SAController:[4,1,1,""]},"paddleslim.common.sa_controller.SAController":{best_tokens:[4,4,1,""],current_tokens:[4,4,1,""],max_reward:[4,4,1,""],next_tokens:[4,2,1,""],update:[4,2,1,""]},"paddleslim.core":{GraphWrapper:[5,1,1,""],OpWrapper:[5,1,1,""],Registry:[5,1,1,""],VarWrapper:[5,1,1,""],graph_wrapper:[5,0,0,"-"],registry:[5,0,0,"-"]},"paddleslim.core.GraphWrapper":{"var":[5,2,1,""],all_parameters:[5,2,1,""],clone:[5,2,1,""],get_param_by_op:[5,2,1,""],infer_shape:[5,2,1,""],is_parameter:[5,2,1,""],is_persistable:[5,2,1,""],next_ops:[5,2,1,""],numel_params:[5,2,1,""],ops:[5,2,1,""],pre_ops:[5,2,1,""],program:[5,2,1,""],update_groups_of_conv:[5,2,1,""],update_param_shape:[5,2,1,""],vars:[5,2,1,""]},"paddleslim.core.OpWrapper":{all_inputs:[5,2,1,""],all_outputs:[5,2,1,""],attr:[5,2,1,""],idx:[5,2,1,""],inputs:[5,2,1,""],is_bwd_op:[5,2,1,""],is_opt_op:[5,2,1,""],outputs:[5,2,1,""],set_attr:[5,2,1,""],type:[5,2,1,""]},"paddleslim.core.Registry":{get:[5,2,1,""],module_dict:[5,4,1,""],name:[5,4,1,""],register:[5,2,1,""]},"paddleslim.core.VarWrapper":{inputs:[5,2,1,""],is_parameter:[5,2,1,""],name:[5,2,1,""],outputs:[5,2,1,""],set_shape:[5,2,1,""],shape:[5,2,1,""]},"paddleslim.core.graph_wrapper":{GraphWrapper:[5,1,1,""],OpWrapper:[5,1,1,""],VarWrapper:[5,1,1,""]},"paddleslim.core.graph_wrapper.GraphWrapper":{"var":[5,2,1,""],all_parameters:[5,2,1,""],clone:[5,2,1,""],get_param_by_op:[5,2,1,""],infer_shape:[5,2,1,""],is_parameter:[5,2,1,""],is_persistable:[5,2,1,""],next_ops:[5,2,1,""],numel_params:[5,2,1,""],ops:[5,2,1,""],pre_ops:[5,2,1,""],program:[5,2,1,""],update_groups_of_conv:[5,2,1,""],update_param_shape:[5,2,1,""],vars:[5,2,1,""]},"paddleslim.core.graph_wrapper.OpWrapper":{all_inputs:[5,2,1,""],all_outputs:[5,2,1,""],attr:[5,2,1,""],idx:[5,2,1,""],inputs:[5,2,1,""],is_bwd_op:[5,2,1,""],is_opt_op:[5,2,1,""],outputs:[5,2,1,""],set_attr:[5,2,1,""],type:[5,2,1,""]},"paddleslim.core.graph_wrapper.VarWrapper":{inputs:[5,2,1,""],is_parameter:[5,2,1,""],name:[5,2,1,""],outputs:[5,2,1,""],set_shape:[5,2,1,""],shape:[5,2,1,""]},"paddleslim.core.registry":{Registry:[5,1,1,""]},"paddleslim.core.registry.Registry":{get:[5,2,1,""],module_dict:[5,4,1,""],name:[5,4,1,""],register:[5,2,1,""]},"paddleslim.dist":{single_distiller:[6,0,0,"-"]},"paddleslim.dist.single_distiller":{fsp_loss:[6,3,1,""],l2_loss:[6,3,1,""],loss:[6,3,1,""],merge:[6,3,1,""],soft_label_loss:[6,3,1,""]},"paddleslim.models":{classification_models:[7,0,0,"-"],image_classification:[7,3,1,""],mobilenet:[7,0,0,"-"],mobilenet_v2:[7,0,0,"-"],resnet:[7,0,0,"-"],util:[7,0,0,"-"]},"paddleslim.models.classification_models":{MobileNet:[7,1,1,""],MobileNetV2:[7,1,1,""],ResNet34:[7,3,1,""],ResNet50:[7,3,1,""]},"paddleslim.models.classification_models.MobileNet":{conv_bn_layer:[7,2,1,""],depthwise_separable:[7,2,1,""],net:[7,2,1,""]},"paddleslim.models.classification_models.MobileNetV2":{conv_bn_layer:[7,2,1,""],inverted_residual_unit:[7,2,1,""],invresi_blocks:[7,2,1,""],net:[7,2,1,""],shortcut:[7,2,1,""]},"paddleslim.models.mobilenet":{MobileNet:[7,1,1,""]},"paddleslim.models.mobilenet.MobileNet":{conv_bn_layer:[7,2,1,""],depthwise_separable:[7,2,1,""],net:[7,2,1,""]},"paddleslim.models.mobilenet_v2":{MobileNetV2:[7,1,1,""],MobileNetV2_scale:[7,3,1,""],MobileNetV2_x1_0:[7,3,1,""],MobileNetV2_x1_5:[7,3,1,""],MobileNetV2_x2_0:[7,3,1,""]},"paddleslim.models.mobilenet_v2.MobileNetV2":{conv_bn_layer:[7,2,1,""],inverted_residual_unit:[7,2,1,""],invresi_blocks:[7,2,1,""],net:[7,2,1,""],shortcut:[7,2,1,""]},"paddleslim.models.resnet":{ResNet101:[7,3,1,""],ResNet152:[7,3,1,""],ResNet34:[7,3,1,""],ResNet50:[7,3,1,""],ResNet:[7,1,1,""]},"paddleslim.models.resnet.ResNet":{basic_block:[7,2,1,""],bottleneck_block:[7,2,1,""],conv_bn_layer:[7,2,1,""],net:[7,2,1,""],shortcut:[7,2,1,""]},"paddleslim.models.util":{image_classification:[7,3,1,""]},"paddleslim.nas":{InceptionABlockSpace:[8,1,1,""],InceptionCBlockSpace:[8,1,1,""],MobileNetV1BlockSpace:[8,1,1,""],MobileNetV1Space:[8,1,1,""],MobileNetV2BlockSpace:[8,1,1,""],MobileNetV2Space:[8,1,1,""],ResNetBlockSpace:[8,1,1,""],ResNetSpace:[8,1,1,""],SANAS:[8,1,1,""],SearchSpaceBase:[8,1,1,""],SearchSpaceFactory:[8,1,1,""],one_shot:[9,0,0,"-"],sa_nas:[8,0,0,"-"],search_space:[10,0,0,"-"]},"paddleslim.nas.InceptionABlockSpace":{init_tokens:[8,2,1,""],range_table:[8,2,1,""],token2arch:[8,2,1,""]},"paddleslim.nas.InceptionCBlockSpace":{init_tokens:[8,2,1,""],range_table:[8,2,1,""],token2arch:[8,2,1,""]},"paddleslim.nas.MobileNetV1BlockSpace":{init_tokens:[8,2,1,""],range_table:[8,2,1,""],token2arch:[8,2,1,""]},"paddleslim.nas.MobileNetV1Space":{init_tokens:[8,2,1,""],range_table:[8,2,1,""],token2arch:[8,2,1,""]},"paddleslim.nas.MobileNetV2BlockSpace":{init_tokens:[8,2,1,""],range_table:[8,2,1,""],token2arch:[8,2,1,""]},"paddleslim.nas.MobileNetV2Space":{init_tokens:[8,2,1,""],range_table:[8,2,1,""],token2arch:[8,2,1,""]},"paddleslim.nas.ResNetBlockSpace":{init_tokens:[8,2,1,""],range_table:[8,2,1,""],token2arch:[8,2,1,""]},"paddleslim.nas.ResNetSpace":{init_tokens:[8,2,1,""],range_table:[8,2,1,""],token2arch:[8,2,1,""]},"paddleslim.nas.SANAS":{current_info:[8,2,1,""],next_archs:[8,2,1,""],reward:[8,2,1,""],tokens2arch:[8,2,1,""]},"paddleslim.nas.SearchSpaceBase":{init_tokens:[8,2,1,""],range_table:[8,2,1,""],super_net:[8,2,1,""],token2arch:[8,2,1,""]},"paddleslim.nas.SearchSpaceFactory":{get_search_space:[8,2,1,""]},"paddleslim.nas.one_shot":{OneShotSearch:[9,3,1,""],OneShotSuperNet:[9,1,1,""],SuperMnasnet:[9,1,1,""],one_shot_nas:[9,0,0,"-"],super_mnasnet:[9,0,0,"-"]},"paddleslim.nas.one_shot.OneShotSuperNet":{forward:[9,2,1,""],init_tokens:[9,2,1,""],range_table:[9,2,1,""]},"paddleslim.nas.one_shot.SuperMnasnet":{get_flops:[9,2,1,""],init_tokens:[9,2,1,""],range_table:[9,2,1,""]},"paddleslim.nas.one_shot.one_shot_nas":{OneShotSearch:[9,3,1,""],OneShotSuperNet:[9,1,1,""]},"paddleslim.nas.one_shot.one_shot_nas.OneShotSuperNet":{forward:[9,2,1,""],init_tokens:[9,2,1,""],range_table:[9,2,1,""]},"paddleslim.nas.one_shot.super_mnasnet":{SuperMnasnet:[9,1,1,""]},"paddleslim.nas.one_shot.super_mnasnet.SuperMnasnet":{get_flops:[9,2,1,""],init_tokens:[9,2,1,""],range_table:[9,2,1,""]},"paddleslim.nas.sa_nas":{SANAS:[8,1,1,""]},"paddleslim.nas.sa_nas.SANAS":{current_info:[8,2,1,""],next_archs:[8,2,1,""],reward:[8,2,1,""],tokens2arch:[8,2,1,""]},"paddleslim.nas.search_space":{InceptionABlockSpace:[10,1,1,""],InceptionCBlockSpace:[10,1,1,""],MobileNetV1BlockSpace:[10,1,1,""],MobileNetV1Space:[10,1,1,""],MobileNetV2BlockSpace:[10,1,1,""],MobileNetV2Space:[10,1,1,""],ResNetBlockSpace:[10,1,1,""],ResNetSpace:[10,1,1,""],SearchSpaceBase:[10,1,1,""],SearchSpaceFactory:[10,1,1,""],base_layer:[10,0,0,"-"],combine_search_space:[10,0,0,"-"],inception_block:[10,0,0,"-"],mobilenet_block:[10,0,0,"-"],mobilenetv1:[10,0,0,"-"],mobilenetv2:[10,0,0,"-"],resnet:[10,0,0,"-"],resnet_block:[10,0,0,"-"],search_space_base:[10,0,0,"-"],search_space_factory:[10,0,0,"-"],search_space_registry:[10,0,0,"-"],utils:[10,0,0,"-"]},"paddleslim.nas.search_space.InceptionABlockSpace":{init_tokens:[10,2,1,""],range_table:[10,2,1,""],token2arch:[10,2,1,""]},"paddleslim.nas.search_space.InceptionCBlockSpace":{init_tokens:[10,2,1,""],range_table:[10,2,1,""],token2arch:[10,2,1,""]},"paddleslim.nas.search_space.MobileNetV1BlockSpace":{init_tokens:[10,2,1,""],range_table:[10,2,1,""],token2arch:[10,2,1,""]},"paddleslim.nas.search_space.MobileNetV1Space":{init_tokens:[10,2,1,""],range_table:[10,2,1,""],token2arch:[10,2,1,""]},"paddleslim.nas.search_space.MobileNetV2BlockSpace":{init_tokens:[10,2,1,""],range_table:[10,2,1,""],token2arch:[10,2,1,""]},"paddleslim.nas.search_space.MobileNetV2Space":{init_tokens:[10,2,1,""],range_table:[10,2,1,""],token2arch:[10,2,1,""]},"paddleslim.nas.search_space.ResNetBlockSpace":{init_tokens:[10,2,1,""],range_table:[10,2,1,""],token2arch:[10,2,1,""]},"paddleslim.nas.search_space.ResNetSpace":{init_tokens:[10,2,1,""],range_table:[10,2,1,""],token2arch:[10,2,1,""]},"paddleslim.nas.search_space.SearchSpaceBase":{init_tokens:[10,2,1,""],range_table:[10,2,1,""],super_net:[10,2,1,""],token2arch:[10,2,1,""]},"paddleslim.nas.search_space.SearchSpaceFactory":{get_search_space:[10,2,1,""]},"paddleslim.nas.search_space.base_layer":{conv_bn_layer:[10,3,1,""]},"paddleslim.nas.search_space.combine_search_space":{CombineSearchSpace:[10,1,1,""]},"paddleslim.nas.search_space.combine_search_space.CombineSearchSpace":{init_tokens:[10,2,1,""],range_table:[10,2,1,""],token2arch:[10,2,1,""]},"paddleslim.nas.search_space.inception_block":{InceptionABlockSpace:[10,1,1,""],InceptionCBlockSpace:[10,1,1,""]},"paddleslim.nas.search_space.inception_block.InceptionABlockSpace":{init_tokens:[10,2,1,""],range_table:[10,2,1,""],token2arch:[10,2,1,""]},"paddleslim.nas.search_space.inception_block.InceptionCBlockSpace":{init_tokens:[10,2,1,""],range_table:[10,2,1,""],token2arch:[10,2,1,""]},"paddleslim.nas.search_space.mobilenet_block":{MobileNetV1BlockSpace:[10,1,1,""],MobileNetV2BlockSpace:[10,1,1,""]},"paddleslim.nas.search_space.mobilenet_block.MobileNetV1BlockSpace":{init_tokens:[10,2,1,""],range_table:[10,2,1,""],token2arch:[10,2,1,""]},"paddleslim.nas.search_space.mobilenet_block.MobileNetV2BlockSpace":{init_tokens:[10,2,1,""],range_table:[10,2,1,""],token2arch:[10,2,1,""]},"paddleslim.nas.search_space.mobilenetv1":{MobileNetV1Space:[10,1,1,""]},"paddleslim.nas.search_space.mobilenetv1.MobileNetV1Space":{init_tokens:[10,2,1,""],range_table:[10,2,1,""],token2arch:[10,2,1,""]},"paddleslim.nas.search_space.mobilenetv2":{MobileNetV2Space:[10,1,1,""]},"paddleslim.nas.search_space.mobilenetv2.MobileNetV2Space":{init_tokens:[10,2,1,""],range_table:[10,2,1,""],token2arch:[10,2,1,""]},"paddleslim.nas.search_space.resnet":{ResNetSpace:[10,1,1,""]},"paddleslim.nas.search_space.resnet.ResNetSpace":{init_tokens:[10,2,1,""],range_table:[10,2,1,""],token2arch:[10,2,1,""]},"paddleslim.nas.search_space.resnet_block":{ResNetBlockSpace:[10,1,1,""]},"paddleslim.nas.search_space.resnet_block.ResNetBlockSpace":{init_tokens:[10,2,1,""],range_table:[10,2,1,""],token2arch:[10,2,1,""]},"paddleslim.nas.search_space.search_space_base":{SearchSpaceBase:[10,1,1,""]},"paddleslim.nas.search_space.search_space_base.SearchSpaceBase":{init_tokens:[10,2,1,""],range_table:[10,2,1,""],super_net:[10,2,1,""],token2arch:[10,2,1,""]},"paddleslim.nas.search_space.search_space_factory":{SearchSpaceFactory:[10,1,1,""]},"paddleslim.nas.search_space.search_space_factory.SearchSpaceFactory":{get_search_space:[10,2,1,""]},"paddleslim.nas.search_space.utils":{check_points:[10,3,1,""],compute_downsample_num:[10,3,1,""],get_random_tokens:[10,3,1,""]},"paddleslim.pantheon":{Student:[11,1,1,""],Teacher:[11,1,1,""],student:[11,0,0,"-"],teacher:[11,0,0,"-"],utils:[11,0,0,"-"]},"paddleslim.pantheon.Student":{get_knowledge_desc:[11,2,1,""],get_knowledge_generator:[11,2,1,""],get_knowledge_qsize:[11,2,1,""],recv:[11,2,1,""],register_teacher:[11,2,1,""],send:[11,2,1,""],start:[11,2,1,""]},"paddleslim.pantheon.Teacher":{dump:[11,2,1,""],recv:[11,2,1,""],send:[11,2,1,""],start:[11,2,1,""],start_knowledge_service:[11,2,1,""]},"paddleslim.pantheon.student":{Student:[11,1,1,""]},"paddleslim.pantheon.student.Student":{get_knowledge_desc:[11,2,1,""],get_knowledge_generator:[11,2,1,""],get_knowledge_qsize:[11,2,1,""],recv:[11,2,1,""],register_teacher:[11,2,1,""],send:[11,2,1,""],start:[11,2,1,""]},"paddleslim.pantheon.teacher":{Teacher:[11,1,1,""]},"paddleslim.pantheon.teacher.Teacher":{dump:[11,2,1,""],recv:[11,2,1,""],send:[11,2,1,""],start:[11,2,1,""],start_knowledge_service:[11,2,1,""]},"paddleslim.pantheon.utils":{EndSignal:[11,1,1,""],StartSignal:[11,1,1,""],SyncSignal:[11,1,1,""],check_ip:[11,3,1,""],convert_dtype:[11,3,1,""]},"paddleslim.prune":{AutoPruner:[12,1,1,""],Pruner:[12,1,1,""],SensitivePruner:[12,1,1,""],auto_pruner:[12,0,0,"-"],conv2d:[12,1,1,""],flops_sensitivity:[12,3,1,""],get_ratios_by_loss:[12,3,1,""],load_model:[12,3,1,""],load_sensitivities:[12,3,1,""],merge_sensitive:[12,3,1,""],prune_io:[12,0,0,"-"],prune_walker:[12,0,0,"-"],pruner:[12,0,0,"-"],save_model:[12,3,1,""],sensitive:[12,0,0,"-"],sensitive_pruner:[12,0,0,"-"],sensitivity:[12,3,1,""]},"paddleslim.prune.AutoPruner":{prune:[12,2,1,""],reward:[12,2,1,""]},"paddleslim.prune.Pruner":{prune:[12,2,1,""]},"paddleslim.prune.SensitivePruner":{get_ratios_by_sensitive:[12,2,1,""],greedy_prune:[12,2,1,""],prune:[12,2,1,""],restore:[12,2,1,""],save_checkpoint:[12,2,1,""]},"paddleslim.prune.auto_pruner":{AutoPruner:[12,1,1,""]},"paddleslim.prune.auto_pruner.AutoPruner":{prune:[12,2,1,""],reward:[12,2,1,""]},"paddleslim.prune.prune_io":{load_model:[12,3,1,""],save_model:[12,3,1,""]},"paddleslim.prune.prune_walker":{conv2d:[12,1,1,""]},"paddleslim.prune.pruner":{Pruner:[12,1,1,""]},"paddleslim.prune.pruner.Pruner":{prune:[12,2,1,""]},"paddleslim.prune.sensitive":{flops_sensitivity:[12,3,1,""],get_ratios_by_loss:[12,3,1,""],load_sensitivities:[12,3,1,""],merge_sensitive:[12,3,1,""],sensitivity:[12,3,1,""]},"paddleslim.prune.sensitive_pruner":{SensitivePruner:[12,1,1,""]},"paddleslim.prune.sensitive_pruner.SensitivePruner":{get_ratios_by_sensitive:[12,2,1,""],greedy_prune:[12,2,1,""],prune:[12,2,1,""],restore:[12,2,1,""],save_checkpoint:[12,2,1,""]},"paddleslim.quant":{quant_embedding:[13,0,0,"-"],quanter:[13,0,0,"-"]},"paddleslim.quant.quant_embedding":{quant_embedding:[13,3,1,""]},"paddleslim.quant.quanter":{convert:[13,3,1,""],quant_aware:[13,3,1,""],quant_post:[13,3,1,""]},paddleslim:{analysis:[3,0,0,"-"],common:[4,0,0,"-"],core:[5,0,0,"-"],dist:[6,0,0,"-"],models:[7,0,0,"-"],nas:[8,0,0,"-"],pantheon:[11,0,0,"-"],prune:[12,0,0,"-"],quant:[13,0,0,"-"],version:[2,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","function","Python function"],"4":["py","attribute","Python attribute"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:function","4":"py:attribute"},terms:{"6\u7248\u672c\u6216\u66f4\u65b0\u7248\u672c":16,"7\u7248\u672c":[20,24],"\u4e0b\u8f7d":18,"\u4e0b\u8f7d\u94fe\u63a5":18,"\u4e0b\u9884\u5b9a\u4e49\u4e86\u7528\u4e8e\u6784\u5efa\u5206\u7c7b\u6a21\u578b\u7684\u65b9\u6cd5":20,"\u4e0d\u4ec5\u5b9e\u73b0\u4e86\u76ee\u524d\u4e3b\u6d41\u7684\u7f51\u7edc\u526a\u679d":17,"\u4e2d\u5b58\u50a8\u7684\u53c2\u6570\u6570\u7ec4\u8fdb\u884c\u88c1\u526a":20,"\u4e2d\u5bf9\u5e94\u5377\u79ef\u5c42\u53c2\u6570\u7684\u5b9a\u4e49":20,"\u4e2d\u6587\u6587\u6863":15,"\u4e3a\u4e86\u5feb\u901f\u6267\u884c\u8be5\u793a\u4f8b":20,"\u4e3a\u4e86\u65b9\u4fbf\u5c55\u793a\u793a\u4f8b":20,"\u4e3b\u8981\u7528\u4e8e\u538b\u7f29\u56fe\u50cf\u9886\u57df\u6a21\u578b":17,"\u4ee3\u7801\u5982\u4e0b":20,"\u4ee3\u7801\u5982\u4e0b\u6240\u793a":20,"\u4ee5\u4e0a\u64cd\u4f5c\u4f1a\u4fee\u6539":20,"\u4ee5\u4e0b\u4ee3\u7801\u6267\u884c\u4e86\u4e00\u4e2a":20,"\u4ee5\u4e0b\u7ae0\u8282\u4f9d\u6b21\u6b21\u4ecb\u7ecd\u6bcf\u4e2a\u6b65\u9aa4\u7684\u5185\u5bb9":[20,24],"\u4ee5\u53ca\u5b8c\u5584\u5bf9nlp\u9886\u57df\u6a21\u578b\u7684\u652f\u6301":17,"\u4f1a\u6dfb\u52a0\u66f4\u591a\u7684\u538b\u7f29\u7b56\u7565":17,"\u5206\u522b\u526a\u638920":20,"\u5206\u6790\u654f\u611f\u5ea6":24,"\u526a\u88c1\u6a21\u578b":24,"\u5305\u5b9a\u4e49\u4e86mnist\u6570\u636e\u7684\u4e0b\u8f7d\u548c\u8bfb\u53d6":20,"\u538b\u7f29\u65b9\u6cd5":18,"\u540c\u65f6\u5bf9":20,"\u548c":20,"\u548c30":20,"\u5728\u540e\u7eed\u7248\u672c\u4e2d":17,"\u5728\u7ebf\u91cf\u5316\u8bad\u7ec3":17,"\u5728paddleslim\u4e2d":17,"\u57fa\u4e8e\u654f\u611f\u5ea6\u7684\u6a21\u578b\u526a\u88c1":17,"\u57fa\u4e8e\u8fdb\u5316\u7b97\u6cd5\u7684\u81ea\u52a8\u6a21\u578b\u526a\u88c1\u4e09\u79cd\u65b9\u5f0f":17,"\u5b89\u88c5\u5386\u53f2\u7248\u672c":16,"\u5b89\u88c5\u5b98\u65b9\u53d1\u5e03\u7684\u6700\u65b0\u7248\u672c":16,"\u5b89\u88c5develop\u7248\u672c":16,"\u5b89\u88c5paddleslim\u524d":16,"\u5b9a\u4e49\u6a21\u578b\u8bc4\u4f30\u65b9\u6cd5":24,"\u5b9a\u4e49\u8f93\u5165\u6570\u636e":24,"\u5e26_vd\u540e\u7f00\u4ee3\u8868\u8be5\u9884\u8bad\u7ec3\u6a21\u578b\u4f7f\u7528\u4e86mixup":18,"\u5e76\u5c06\u8f93\u5165\u5927\u5c0f\u8bbe\u7f6e\u4e3a":20,"\u6211\u4eec\u5728":20,"\u6211\u4eec\u8fd9\u91cc\u5bf9\u53c2\u6570\u540d\u4e3a":20,"\u6211\u4eec\u9009\u53d6\u7b80\u5355\u7684mnist\u6570\u636e":20,"\u6267\u884c\u4ee5\u4e0b\u4ee3\u7801\u6784\u5efa\u5206\u7c7b\u6a21\u578b":20,"\u652f\u6301":17,"\u652f\u6301\u57fa\u4e8e\u8fdb\u5316\u7b97\u6cd5\u7684\u8f7b\u91cf\u795e\u7ecf\u7f51\u7edc\u7ed3\u6784\u81ea\u52a8\u641c\u7d22":17,"\u652f\u6301\u591a\u5e73\u53f0\u6a21\u578b\u5ef6\u65f6\u8bc4\u4f30":17,"\u652f\u6301\u5bf9\u6743\u91cd\u5168\u5c40\u91cf\u5316\u548cchannel":17,"\u652f\u6301\u901a\u9053\u5747\u5300\u6a21\u578b\u526a\u88c1":17,"\u6570\u636e\u96c6":18,"\u662f\u4e3a\u4e86\u7b80\u5316\u793a\u4f8b\u800c\u5c01\u88c5\u9884\u5b9a\u4e49\u7684\u4e00\u7cfb\u5217\u65b9\u6cd5":20,"\u6784\u5efa\u6a21\u578b":[20,24],"\u67e5\u770b\u53ef\u5b89\u88c5\u5386\u53f2\u7248\u672c":16,"\u6a21\u578b":18,"\u6a21\u578b\u4f53\u79ef":18,"\u6a21\u578b\u526a\u88c1":17,"\u6a21\u578b\u7ed3\u6784\u7684\u5b9a\u4e49":20,"\u6bd4\u5982":20,"\u6ce8\u610f":20,"\u7136\u540e\u6309\u4ee5\u4e0b\u65b9\u5f0f\u5bfc\u5165paddle\u548cpaddleslim":[20,24],"\u7684\u5377\u79ef\u5c42\u8fdb\u884c\u526a\u88c1":20,"\u7684\u8bad\u7ec3":20,"\u7684\u901a\u9053\u6570":20,"\u786c\u4ef6\u5ef6\u65f6\u7ea6\u675f":17,"\u79bb\u7ebf\u91cf\u5316":17,"\u83b7\u53d6\u5f85\u5206\u6790\u5377\u79ef\u53c2\u6570\u540d\u79f0":24,"\u84b8\u998f":17,"\u84b8\u998f\u4e09\u79cd\u538b\u7f29\u7b56\u7565":17,"\u8bad\u7ec3\u6a21\u578b":24,"\u8be5\u6559\u7a0b\u4ee5\u56fe\u50cf\u5206\u7c7b\u6a21\u578bmobilenetv1\u4e3a\u4f8b":[20,24],"\u8be5\u793a\u4f8b\u5305\u542b\u4ee5\u4e0b\u6b65\u9aa4":[20,24],"\u8be5\u7ae0\u8282\u6784\u9020\u4e00\u4e2a\u7528\u4e8e\u5bf9mnist\u6570\u636e\u8fdb\u884c\u5206\u7c7b\u7684\u5206\u7c7b\u6a21\u578b":20,"\u8bf4\u660e\u5982\u4f55\u5feb\u901f\u4f7f\u7528":[20,24],"\u8bf7\u70b9\u51fb":16,"\u8bf7\u786e\u8ba4\u5df2\u6b63\u786e\u5b89\u88c5paddl":[20,24],"\u8bf7\u786e\u8ba4\u5df2\u6b63\u786e\u5b89\u88c5paddle1":16,"\u8f7b\u91cf\u795e\u7ecf\u7f51\u7edc\u7ed3\u6784\u81ea\u52a8\u641c\u7d22":17,"\u8f93\u5165320":18,"\u8f93\u5165416":18,"\u8f93\u5165608":18,"\u8f93\u5165\u5c3a\u5bf8":18,"\u8f93\u51fa\u7c7b\u522b\u6570\u4e3a10":20,"\u8fd8\u5b9e\u73b0\u4e86\u8d85\u53c2\u6570\u641c\u7d22\u548c\u5c0f\u6a21\u578b\u7f51\u7edc\u7ed3\u6784\u641c\u7d22\u529f\u80fd":17,"\u9009\u7528":20,"\u91cf\u5316":17,"\u91cf\u5316\u8bad\u7ec3":17,"abstract":[3,4],"class":[3,4,5,7,8,9,10,11,12],"default":[3,4,6,9,11,12,13,21],"final":[9,21,22],"float":[3,4,5,6,8,12,21],"function":[4,5,6,8,9,10,13,21,22],"imagenet1000\u7c7b":18,"import":[4,20,24],"int":[3,4,5,8,9,10,11,13],"mixup\u76f8\u5173\u4ecb\u7ecd\u53c2\u8003":18,"models\u4e0b\u7684api\u5e76\u975epaddleslim\u5e38\u89c4api":20,"new":[4,5],"paddle\u5b89\u88c5\u6559\u7a0b":16,"paddle\u5b89\u88c5\u8bf7\u53c2\u8003":16,"paddle\u6846\u67b6\u7684":20,"paddleslim\u4f9d\u8d56paddle1":[20,24],"paddleslim\u662fpaddlepaddle\u6846\u67b6\u7684\u4e00\u4e2a\u5b50\u6a21\u5757":17,"paddleslim\u7684\u5377\u79ef\u901a\u9053\u526a\u88c1\u63a5\u53e3":20,"paddleslim\u7684\u654f\u611f\u5ea6\u5206\u6790\u63a5\u53e3":24,"program\u7684\u6784\u5efa\u7b49":20,"public":11,"return":[3,4,5,6,8,9,10,11,12,13],"static":11,"super":[8,9,10],"true":[3,7,8,10,11,12,13,20,21,22],"var":[5,6,13,21,22],"while":[8,9,12],"wise\u91cf\u5316":17,And:12,For:4,NAS:[8,10,17,18],The:[3,4,5,6,8,9,10,11,12,13,21,22],Then:[21,22],There:22,_scope:6,aadvanc:15,abs:18,abs_max:13,acc1:[20,21,22],acc5:[20,21,22],acc:18,accept:9,accord:[4,5,8,10,21],accuraci:[12,13,21,22],accuracy_loss:12,act:[7,10],activ:[10,22],add:[3,6,13,21],added:[6,11],address:[4,11],after:[4,5,22],agent:4,algo:13,all:[3,4,5,6,8,11,13],all_input:5,all_output:5,all_paramet:5,alreadi:11,also:12,analysi:[0,1,2,20],ani:[5,11],anneal:4,api:[13,15,21,22],append:[21,22],appli:13,arch:[8,9,10],architectur:[4,8,10],arg:[4,8,10],argument:[9,11],arrai:[21,22],arxiv:18,asctim:4,assign:11,attr:5,attribut:5,auto:18,auto_prun:[1,2],autoprun:12,awar:[17,19],backup:12,backward:5,base:[3,4,5,8,9,10,11,12],base_lay:[2,8],baselin:18,basic_block:7,basicconfig:4,batch:[10,11,13,20,21,22],batch_gener:11,batch_num:[13,22],batch_siz:[11,13,20,21,22],becaus:[21,22],befor:[6,13],best:[8,9],best_token:4,between:6,beyond:18,bit:13,blazefac:18,block:6,block_mask:[8,10],block_num:[8,10],bool:[3,5,8,10,11,12,13],bottleneck_block:7,box:18,buf_siz:11,buffer:11,build:10,cach:4,cache_dir:13,cache_path:4,cached_id:4,cached_read:[1,2],calcul:[3,12,13,22],calibr:[13,22],call:[9,11,21,22],callback:9,can:[5,11,13,21],cann:12,cannot:[13,21],capac:11,carri:11,ch_out:7,chang:[13,21],change_depth:7,channel:[7,8,9,10],check:[4,13],check_ip:11,check_point:10,checkpoint:[4,12],cityscap:18,class_dim:7,class_num:7,classif:[19,23],classification_model:[1,2],client:4,client_nam:4,client_num:4,clip:13,clone:[5,16],close:[4,21,22],cls:5,coco:18,com:16,combin:[6,10],combine_search_spac:[2,8],combinesearchspac:10,command:11,common:[0,1,2,11],comput:9,compute_downsample_num:10,config:[8,10,11,13],config_list:[8,10],configur:21,consid:12,consist:6,constrain:[8,10],constrain_func:4,constraint:4,construct:[21,22],contain:[9,11,21,22],control:[1,2,8,10,12],control_token:4,controller_cli:[1,2],controller_serv:[1,2],controllercli:4,controllerserv:4,conv1:7,conv1_nam:7,conv2_1_sep_weight:20,conv2_2_sep_weight:20,conv2d:[12,13],conv_bn_lay:[7,10],conveni:[21,22],convert:[8,13,21],convert_dtyp:11,convlut:3,convolut:[3,5,10],core:[0,1,2],core_avx:6,correctli:[21,22],correspond:11,count:[3,10],cpu:[11,13],cpu_num:11,cpuplac:[6,13,20,21,22],creat:[8,10],criterion:12,cuda:13,cuda_visible_devic:11,cudaplac:[6,13],cudnn:10,current:[3,4,5,8,9,10,12],current_info:8,current_token:4,cut:12,data:[4,11,13,20],data_map_map:6,data_name_map:6,data_residu:7,datafeed:[20,21,22],dataload:[11,13],dataset:[4,20,21,22],dcn:18,deeplabv3:18,def:[21,22],defalut:[12,13],default_main_program:6,defin:[3,4,6,8,9,11,21,22],delimit:3,denot:11,depend:[21,22],depth:[8,10],depthwise_conv2d:13,depthwise_separ:7,dequant:[13,21],describ:6,descript:11,detail:3,determin:11,devic:[3,6,11,12,13],dict:[3,8,11,12,13],dictionari:[8,11],differ:11,dimens:6,direct:13,directori:12,dirnam:[12,21,22],disabl:4,disk:13,dist:[0,1,2],distil:[6,18],divergenc:13,divid:6,document:15,doe:[21,22],doesn:11,doing:22,down:12,download:[21,22],downsampl:[8,10],drop:11,drop_last:[11,20,21,22],dtype:[11,13,21],dump:11,dygraph:9,dynam:[9,11],each:[3,8,9,10,11,12],easi:18,element:[5,12],empir:18,end:11,endsign:11,engin:3,enhanc:13,environ:11,epoch:[20,21,22],eval_func:[9,12],eval_program:12,eval_read:4,evalu:[3,4,8,10,12],everi:[9,13],evolutionari:4,evolutionarycontrol:4,exampl:[4,21,22],except:6,exe:[11,12,20,21,22],execut:[9,11],executor:[11,13,21,22],expansion_factor:[7,8,10],face:18,factor:13,failur:8,fake:13,fals:[3,5,7,8,9,11,12,13,20,21],fast:18,fc_name:7,feed:[11,20,21,22],feed_list:11,feed_target_nam:22,feeded_var_nam:[21,22],fetch:11,fetch_list:[20,21,22],fetch_target:22,file:[3,4,5,11,12,13],filenam:13,filesystem:12,filter:[5,10,12],filter_num1:[8,10],filter_num2:[8,10],filter_num:[8,10],filter_s:[7,10],finetun:[13,21],finish:11,first:[8,10,21,22],flags_selected_gpu:11,float32:[13,21],float_prog:21,flop:[1,2,12,17,20],flops_sensit:12,fluid:[3,5,6,9,11,12,13,20,21,22,24],fmt:4,follow:[8,10,11,21,22],for_test:[5,13,21],formal:[21,22],format:[3,4,9,11,12,20,21],forth:[8,10],forward:[9,22],fp32:13,framework:[5,21,22],freez:13,freezed_program:13,freezed_program_int8:13,from:[4,5,6,11,12,21],fsp:6,fsp_loss:6,func:11,gener:[4,9,11,12,13,22],get:[3,4,5,8,9,10,11,12,13,21,22],get_flop:9,get_knowledge_desc:11,get_knowledge_gener:11,get_knowledge_qs:11,get_logg:4,get_param_by_op:5,get_random_token:10,get_ratios_by_loss:12,get_ratios_by_sensit:12,get_search_spac:[8,10],gflop:18,git:16,github:16,given:[3,4,5,12],global_block:[21,22],global_scop:[13,20],gpu:[11,18],graph:[3,5,8,9,10,11,12],graph_wrapp:[1,2],graphwrapp:5,grapwrapp:3,greedy_prun:12,group:[5,7,10,12],handl:4,hard:18,have:[21,22],head_num:[8,10],how:[21,22],http:[16,18],hyperparamet:4,ids:11,idx:[5,6],if_act:7,ifshortcut:7,ignor:11,imag:[18,19,23],image_classif:[7,20,21,22],image_shap:7,in_address:11,in_c:7,in_nod:5,in_path:11,inception_block:[2,8],inceptionablockspac:[8,10],inceptioncblockspac:[8,10],includ:[5,8,11],index:[8,10],indic:9,infer:[3,11,13,21,22],infer_shap:5,inference_model:[21,22],inferfac:[21,22],inform:[4,8,12],init:[8,9,10],init_ratio:12,init_temperatur:[4,8,12],init_token:[4,8,9,10],initi:[8,10],input:[5,6,7,9,10,11,13,20],input_channel:9,input_s:[8,10],insert:13,instal:[15,21,22],instanc:[5,9],int8:[13,21],int8_prog:21,integ:9,introduct:15,inverted_residual_unit:7,invresi_block:7,irgraph:5,is_bwd_op:5,is_first:7,is_full_quant:13,is_opt_op:5,is_paramet:5,is_persist:5,is_serv:[8,12],is_use_cache_fil:13,iter:[4,21,22],its:11,just:[3,8,10],kei:[3,4,5,8,10,11,12,13],kernel_s:[8,10],knowledg:11,kwarg:[4,6],l1_norm:12,l2_loss:6,label:6,last:[4,11],latenc:[1,2],latencyevalu:3,latest:[4,12],launch:11,layer:[3,5,7,8,9,10],lazi:12,learn:21,least:9,less:[11,12],level:[4,11],levelnam:4,light:17,limit:11,line:[8,10],list:[4,5,8,9,10,11,12,13],lite:[13,18,21],load:[5,12,13,21,22],load_checkpoint:8,load_inference_model:22,load_model:12,load_sensit:12,local:4,lock:[1,2],lod:11,log:4,log_help:[1,2],logger:4,lookup_t:13,loss:[6,12,20,21,22],loss_func:6,main_program:[21,22],make:[21,22],map:6,max:12,max_client_num:[4,12],max_ratio:12,max_reward:4,max_try_tim:[4,12],max_valu:9,maximum:[9,11],mean:[4,8,9,10,11,12,21,22],medium:18,meet:4,memori:13,merg:[6,11,12],merge_sensit:12,merge_strategi:11,messag:4,method:[3,4,11,13,21,22],min_ratio:12,min_valu:9,mini:22,minim:18,minimum:9,miou:18,mixup:18,mnist:[20,21,22],mobilenet:[1,2,18,20,21,22],mobilenet_block:[2,8],mobilenet_v2:[1,2],mobilenetv1:[2,8,18,20,21,22],mobilenetv1blockspac:[8,10],mobilenetv1spac:[8,10],mobilenetv2:[2,7,8,18],mobilenetv2_scal:7,mobilenetv2_x1_0:7,mobilenetv2_x1_5:7,mobilenetv2_x2_0:7,mobilenetv2blockspac:[8,10],mobilenetv2spac:[8,10],mode:[3,9,11],model:[0,1,2,3,6,8,9,10,11,12,13,15,19,23],model_dir:[13,22],model_filenam:13,model_s:[1,2],modifi:12,modul:1,module_dict:5,monitor:11,more:13,mul:[3,13],multi:10,multipl:11,must:[12,13],name:[3,4,5,6,7,8,9,10,11,12,13,18,21,22],name_prefix:6,name_scop:9,nas:[0,1,2],nas_checkpoint:8,necessari:11,need:[21,22],net:[7,9],net_arch:[8,10],network:[8,9,10,12],neural:[4,8,10],next:[4,5,8],next_arch:8,next_op:5,next_token:4,node:[5,11],none:[4,5,7,8,9,10,11,12,13],normal:10,note:[18,21,22],now:11,num_filt:[7,10],num_filters1:7,num_filters2:7,num_group:[7,10],num_in_filt:7,number:[3,5,10,11,13],numel_param:5,numpi:[21,22],obj365_pretrain:18,object:[3,4,5,6,8,10,11,12],obtain:13,offlin:11,onc:11,one:[8,9,10,11,21,22],one_shot:[0,2,8],one_shot_na:[2,8],oneshotsearch:9,oneshotsupernet:9,onli:[3,11,12,13,21,22],onlin:11,only_conv:3,only_graph:12,oper:[3,5,21],ops:[5,13],optim:5,option:13,opwrapp:5,order:[11,13,21,22],org:[16,18],other:[6,11],otherwis:[3,9],out:11,out_channel:9,out_nod:5,out_path:11,out_port:11,output:[4,5,9,10,11,20,21,22],output_s:[8,10],packag:[0,1],pad:[7,10],paddl:[4,5,6,9,13,20,21,22,24],paddle1:[21,22],paddlepaddl:16,paddleslim:[0,16,20,21,22,24],pair:11,pantheon:[0,1,2],param:[4,5,6,8,9,10,12,13,20],param_backup:12,param_nam:12,param_shape_backup:12,paramat:12,paramet:[3,4,5,6,9,11,12,13,21,22],params_filenam:13,params_nam:13,parent_idx:6,pars:11,partial:4,pasacl:18,pascal:18,path:[3,4,11,13],percent:12,perform:[3,9,21,22],persist:5,pip:16,place:[6,12,13,20,21],pleas:[21,22],plu:11,point:10,pop:11,port:[4,11],posit:9,post:[13,17,19],pre:[21,22],pre_op:5,precis:13,predict:11,prefix:6,prefix_nam:7,preform:[21,22],previou:5,print:[20,21,22],process:[21,22],prog:[21,22],program:[3,5,6,11,12,13,21,22],provid:13,prune:[0,1,2,5,17,19,23],prune_io:[1,2],prune_walk:[1,2],pruned_flop:12,pruned_flops_r:12,pruned_lat:12,pruned_param:12,pruned_program:20,pruned_ratio:12,pruner:[1,2,20],prunework:12,push:11,pypi:16,pyread:11,python:[11,13,16],qsize:11,quant:[0,1,2,21,22],quant_awar:[13,18,21],quant_embed:[1,2],quant_post:[13,18,22],quant_post_model:22,quant_post_prog:22,quant_program:21,quanter:[1,2],quantiz:[13,19],quantizable_op_typ:13,quantize_bit:13,quantize_model_path:[13,22],quantize_typ:13,queue:11,quick:15,r50:18,rang:[8,9,10,13,21],range_t:[4,8,9,10],rate:[4,21],ratio:[12,20],reach:[11,21,22],read:[21,22],reader:[4,11,20,21,22],reader_config:11,real:11,reason:[21,22],receiv:11,reciev:11,record:3,recv:11,reduce_r:[4,8,12],regist:[5,11],register_teach:11,registr:11,registri:[1,2],relu:7,repeat:[8,10,11],repeat_num:[8,10],repeat_tim:9,repres:[6,8,9,10],request:4,request_current_info:4,res:[21,22],reset:4,resnet101:[7,18],resnet152:7,resnet34:[7,18],resnet50:[7,18],resnet50_vd1:18,resnet50_vd:18,resnet:[1,2,8],resnet_block:[2,8],resnetblockspac:[8,10],resnetspac:[8,10],respect:11,restor:12,result:[21,22],reward:[4,8,12],risk:18,rtype:[5,8,9],run:[4,6,11,13,20,21,22],sa_control:[1,2],sa_na:[1,2],sacontrol:4,same:[4,10,11,13],sampl:[4,11,13,22],sample_gener:[11,13,22],sample_list_gener:11,sampled_r:4,sana:8,save:[12,13,22],save_checkpoint:[8,12],save_inference_model:[13,21,22],save_int8:[13,21],save_model:12,scale:[7,8,10,13,22],schema:11,scnn:18,scope:[5,6,9,12,13],score:[8,12],search:[4,8,9,10,12],search_spac:[0,2,8],search_space_bas:[2,8],search_space_factori:[2,8],search_space_registri:[2,8],search_step:[4,8,9,12],searchspacebas:[8,10],searchspacefactori:[8,10],second:6,section:[21,22],see:11,select:[21,22],self:[6,8,10],send:11,sensit:[1,2,18,23],sensitive_prun:[1,2],sensitiveprun:12,sensitivities_fil:12,sent:11,separ:13,serv:11,server:4,server_addr:[8,12],server_ip:4,server_port:4,servic:11,set:[4,5,9,11,12,13],set_attr:5,set_shap:5,sett:4,setup:16,shape:[5,11,12],share:11,shortcut:7,shot:[8,9,10],should:[4,6,9,11,13,21],show:[21,22],simpl:16,simplifi:[21,22],simul:4,singl:[12,13],single_distil:[1,2],size:[3,10,11,13,21,22],slim:[5,20,21,22,24],small:[21,22],smaller:21,socket:4,soft:6,soft_label_loss:6,softmax:6,solut:4,some:[3,5],some_batch_genr:11,some_sample_gener:11,some_sample_list_gener:11,sourc:4,space:[4,8,9,10],special:5,specifi:11,speed:[21,22],stage:9,start:[4,11,15],start_knowledge_servic:11,startsign:11,statu:4,step:[4,9,12],store:[12,13],str:[3,4,5,6,9,10,11,12,13],strategi:[8,9,10,11],stride:[7,9,10],string:2,structur:[21,22],student:[1,2,6,18],student_feature_map:6,student_program:6,student_temperatur:6,student_var1:6,student_var1_nam:6,student_var2:6,student_var2_nam:6,student_var:6,student_var_nam:6,sub:9,submodul:1,subpackag:1,successfulli:8,suitabl:21,sum:11,super_mnasnet:[2,8],super_net:[8,10],supermnasnet:9,support:[11,13],sychron:11,synchron:11,syncsign:11,system:4,tabl:[8,9,10],table_fil:3,tablelatencyevalu:3,target:[3,5,12],target_var:[21,22],task:4,teacher:[1,2,6,18],teacher_:6,teacher_feature_map:6,teacher_id:11,teacher_program:6,teacher_temperatur:6,teacher_var1:6,teacher_var1_nam:6,teacher_var2:6,teacher_var2_nam:6,teacher_var:6,teacher_var_nam:6,temp:13,temp_post_train:13,temperatur:6,tensor:[5,13],test:13,test_read:[21,22],than:[11,12],thei:[21,22],them:[4,11],thi:[5,6,8,10,11,13,21,22],thread:11,three:11,threshold:[12,13],time:[8,10,11,13],token2arch:[8,10],token:[4,8,9,10,12],tokens2arch:8,top1:[21,22],top5:[21,22],top:18,topk:12,total:[3,9],train:[8,9,10,12,13,17,19,20],train_feed:[20,21,22],train_program:[12,20,21,22],train_read:[20,21,22],trainabl:13,trans1:18,transfer:11,trial:11,tupl:[3,9,10,12],tutori:[15,21,22],type:[3,4,5,6,8,9,10,11,12,13],under:13,uniform:[6,17,18],uniqu:11,unlock:4,unrel:11,until:11,updat:[4,5,8],update_groups_of_conv:5,update_param_shap:5,use:[4,5,10,13,21,22],use_auxhead:9,use_cudnn:[7,10],use_gpu:[7,20,21,22],used:[3,4,5,6,8,9,10,11,12,13,21],user:[6,13],uses:13,using:[4,13,21,22],util:[1,2,8,13],val_program:[20,21,22],val_quant_program:21,valid:5,valu:[3,5,8,9,11,12,13],varaibl:9,variabl:[5,6,9,10,11,12,13],varibal:5,varwrapp:5,version:[1,6],visit:12,voc:18,wait:11,weight:[12,13],well:[21,22],when:[11,12,13,21,22],whenev:11,whether:[3,4,5,10,11,12],which:[6,8,9,10,13,21,22],whose:[3,9,11,12],wider:18,without:[4,21,22],wrapper:[4,5,11],writer:11,wrote:11,yet:11,yolov3:18,you:[21,22],zero:12,zoo:15},titles:["API Documents","paddleslim","paddleslim package","paddleslim.analysis package","paddleslim.common package","paddleslim.core package","paddleslim.dist package","paddleslim.models package","paddleslim.nas package","paddleslim.nas.one_shot package","paddleslim.nas.search_space package","paddleslim.pantheon package","paddleslim.prune package","paddleslim.quant package","\u4e2d\u6587\u6587\u6863","Welcome to use PaddleSlim.","Install","Introduction","Model Zoo","Quick Start","Pruning of image classification model - quick start","Training-aware Quantization of image classification model - quick start","Post-training Quantization of image classification model - quick start","Aadvanced Tutorials","Pruning of image classification model - sensitivity"],titleterms:{"\u4e2d\u6587\u6587\u6863":14,"\u526a\u88c1":[18,20],"\u526a\u88c1\u5377\u79ef\u5c42\u901a\u9053":20,"\u529f\u80fd":17,"\u56fe\u50cf\u5206\u5272":18,"\u56fe\u8c61\u5206\u7c7b":18,"\u5b9a\u4e49\u8f93\u5165\u6570\u636e":20,"\u5bfc\u5165\u4f9d\u8d56":[20,24],"\u6267\u884c\u8bad\u7ec3":20,"\u6784\u5efa\u7f51\u7edc":20,"\u76ee\u6807\u68c0\u6d4b":18,"\u84b8\u998f":18,"\u8ba1\u7b97\u526a\u88c1\u4e4b\u524d\u7684flop":20,"\u8ba1\u7b97\u526a\u88c1\u4e4b\u540e\u7684flop":20,"\u8bad\u7ec3\u526a\u88c1\u540e\u7684\u6a21\u578b":20,"\u91cf\u5316":18,"import":[21,22],aadvanc:23,after:21,analysi:3,api:0,architectur:[21,22],auto_prun:12,awar:21,base_lay:10,cached_read:4,classif:[20,21,22,24],classification_model:7,combine_search_spac:10,common:4,control:4,controller_cli:4,controller_serv:4,core:5,data:[21,22],definit:[21,22],dist:6,document:0,flop:3,graph_wrapp:5,imag:[20,21,22,24],inception_block:10,input:[21,22],instal:16,introduct:17,latenc:3,lock:4,log_help:4,mobilenet:7,mobilenet_block:10,mobilenet_v2:7,mobilenetv1:10,mobilenetv2:10,model:[7,18,20,21,22,24],model_s:3,modul:[2,3,4,5,6,7,8,9,10,11,12,13],nas:[8,9,10],necessari:[21,22],normal:[21,22],one_shot:9,one_shot_na:9,packag:[2,3,4,5,6,7,8,9,10,11,12,13],paddleslim:[1,2,3,4,5,6,7,8,9,10,11,12,13,15],pantheon:11,post:22,prune:[12,20,24],prune_io:12,prune_walk:12,pruner:12,quant:13,quant_embed:13,quanter:13,quantiz:[21,22],quick:[19,20,21,22],registri:5,resnet:[7,10],resnet_block:10,sa_control:4,sa_na:8,save:21,search_spac:10,search_space_bas:10,search_space_factori:10,search_space_registri:10,sensit:[12,24],sensitive_prun:12,single_distil:6,start:[19,20,21,22],student:11,submodul:[2,3,4,5,6,7,8,9,10,11,12,13],subpackag:[2,8],super_mnasnet:9,teacher:11,test:[21,22],train:[21,22],tutori:23,use:15,util:[7,10,11],version:2,welcom:15,zoo:18}})