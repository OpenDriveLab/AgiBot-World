import copy

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

from go1.internvl.model.internlm2.configuration_internlm2 import InternLM2Config
from go1.internvl.model.internvl_chat.configuration_intern_vit import InternVisionConfig

from .configuration_action_expert import ActionExpertConfig, LatentPlannerConfig

logger = logging.get_logger(__name__)


class GO1ModelConfig(PretrainedConfig):
    model_type = "go1"
    is_composition = True

    def __init__(
        self,
        vision_config=None,
        llm_config=None,
        action_config=None,
        latent_planner_config=None,
        noise_scheduler_config=None,
        use_backbone_lora=0,
        use_llm_lora=0,
        pad2square=False,
        select_layer=-1,
        force_image_size=None,
        downsample_ratio=0.5,
        template=None,
        dynamic_image_size=False,
        use_thumbnail=False,
        ps_version="v1",
        min_dynamic_patch=1,
        max_dynamic_patch=6,
        latent_planning=False,
        norm=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if vision_config is None:
            vision_config = {}
            logger.info("vision_config is None. Initializing the InternVisionConfig with default values.")

        if llm_config is None:
            llm_config = {}
            logger.info("llm_config is None. Initializing the LlamaConfig config with default values (`LlamaConfig`).")

        if action_config is None:
            action_config = {}
            logger.info(
                "action_config is None. Initializing the LlamaConfig(Action Expert) config with default values (`LlamaConfig`)."
            )

        if latent_planner_config is None:
            latent_planner_config = {}
            logger.info(
                "action_config is None. Initializing the LlamaConfig(mediate Action Expert) config with default values (`LlamaConfig`)."
            )

        if noise_scheduler_config is None:
            noise_scheduler_config = {}
            logger.info("noise_scheduler_config is None. Initializing the Schedulur config with `None`.")

        self.vision_config = InternVisionConfig(**vision_config)
        try:
            if "InternLM2ForCausalLM" in llm_config["architectures"][0]:
                self.llm_config = InternLM2Config(**llm_config)
        except:
            self.llm_config = InternLM2Config()
        try:
            if action_config["architectures"][0] == "ActionExpertModel":
                self.action_config = ActionExpertConfig(**action_config)
        except:
            self.action_config = ActionExpertConfig()
        try:
            if latent_planner_config["architectures"][0] == "ActionExpertModel":
                self.latent_planner_config = LatentPlannerConfig(**latent_planner_config)
        except:
            self.latent_planner_config = LatentPlannerConfig()

        self.noise_scheduler_config = noise_scheduler_config
        self.use_backbone_lora = use_backbone_lora
        self.use_llm_lora = use_llm_lora
        self.pad2square = pad2square
        self.select_layer = select_layer
        self.force_image_size = force_image_size
        self.downsample_ratio = downsample_ratio
        self.template = template
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.ps_version = ps_version  # pixel shuffle version
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.norm = norm
        self.latent_planning = latent_planning
        self.action_chunk_size = self.action_config.action_chunk_size
        self.initializer_range = 0.02

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["vision_config"] = self.vision_config.to_dict()
        output["llm_config"] = self.llm_config.to_dict()
        output["action_config"] = self.action_config.to_dict()
        output["latent_planner_config"] = self.latent_planner_config.to_dict()
        output["model_type"] = self.__class__.model_type
        output["use_backbone_lora"] = self.use_backbone_lora
        output["use_llm_lora"] = self.use_llm_lora
        output["pad2square"] = self.pad2square
        output["select_layer"] = self.select_layer
        output["force_image_size"] = self.force_image_size
        output["downsample_ratio"] = self.downsample_ratio
        output["template"] = self.template
        output["dynamic_image_size"] = self.dynamic_image_size
        output["use_thumbnail"] = self.use_thumbnail
        output["ps_version"] = self.ps_version
        output["min_dynamic_patch"] = self.min_dynamic_patch
        output["max_dynamic_patch"] = self.max_dynamic_patch
        output["norm"] = self.norm
        output["latent_planning"] = self.latent_planning
        output["action_chunk_size"] = self.action_chunk_size
        return output
