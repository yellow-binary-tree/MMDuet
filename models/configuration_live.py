
from transformers import PretrainedConfig

class LiveConfigMixin(PretrainedConfig):
    def __init__(self, *, vision_pretrained: str = None,
        frame_resolution: int = None, frame_token_cls: bool = None, frame_token_pooled: list[int] = None, frame_num_tokens: int = None,
        v_placeholder: str = '<v>', v_placeholder_id: int = None,
        stream_loss_weight: float = 1.0, vision_hidden_size=1024, **kwargs
    ):
        super().__init__(**kwargs)
        self.vision_pretrained = vision_pretrained
        self.frame_resolution = frame_resolution
        self.frame_token_cls = frame_token_cls
        self.frame_token_pooled = frame_token_pooled
        self.frame_num_tokens = frame_num_tokens
        self.vision_hidden_size = vision_hidden_size
        self.stream_loss_weight = stream_loss_weight
        self.v_placeholder = v_placeholder
        self.v_placeholder_id = v_placeholder_id


class VideoHeadLiveConfigMixin(PretrainedConfig):
    def __init__(self, *, vision_pretrained: str = None,
        frame_resolution: int = None, frame_token_cls: bool = None, frame_token_pooled: list[int] = None, frame_num_tokens: int = None,
        v_placeholder: str = '<v>', v_placeholder_id: int = None, vision_hidden_size=1024, **kwargs
    ):
        super().__init__(**kwargs)
        self.vision_pretrained = vision_pretrained
        self.frame_resolution = frame_resolution
        self.frame_token_cls = frame_token_cls
        self.frame_token_pooled = frame_token_pooled
        self.frame_num_tokens = frame_num_tokens
        self.vision_hidden_size = vision_hidden_size

        self.v_placeholder = v_placeholder
        self.v_placeholder_id = v_placeholder_id
