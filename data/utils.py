import random, torch, tqdm, os, subprocess, torchvision, pathlib, submitit, math
import cv2
from itertools import takewhile
try:
    torchvision.set_video_backend('video_reader')
except:
    pass
from transformers import AutoModel
from torchvision.transforms.functional import to_pil_image, normalize


def reformat_example_for_debug(data):
    if isinstance(data, torch.Tensor) and len(data.size()) > 1:
        return data.size()
    if isinstance(data, dict):
        return {k: reformat_example_for_debug(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [reformat_example_for_debug(v) for v in data]
    return data


class DictWithTo(dict):
    def to(self, *args, **kwargs):
        return self

def inverse_preprocess_to_pil_images(frames: torch.Tensor, mean: list, std: list):
    frames = normalize(frames, mean=tuple(-m / s for m, s in zip(mean, std)), std=tuple(1.0 / s for s in std))
    frames = (frames * 255).to(torch.uint8)
    return list(map(to_pil_image, frames))

def rand_bool():
    return bool(random.getrandbits(1))

def case_connect(prefix: str, suffix: str):
    if not prefix:
        return suffix[0].upper() + suffix[1:]
    if not suffix:
        return prefix
    if prefix[-1] == ',' or prefix[-1] == ':':
        return prefix + ' ' + suffix[0].lower() + suffix[1:]
    return prefix + ' ' + suffix[0].upper() + suffix[1:]

def batch_temporal_iou(sequences1: torch.Tensor, sequences2: torch.Tensor):
    area1 = sequences1[:, 1] - sequences1[:, 0]
    area2 = sequences2[:, 1] - sequences2[:, 0]
    l = torch.maximum(sequences1[:,None,0], sequences2[:,0])
    r = torch.minimum(sequences1[:,None,1], sequences2[:,1])
    inter = (r - l).clamp(min=0)
    union = area1[:, None] + area2 - inter
    iou = inter / union
    return iou

def temporal_iou(region1, region2):
    area1 = region1[1] - region1[0]
    area2 = region2[1] - region2[0]
    l = max(region1[0], region2[0])
    r = min(region1[1], region2[1])
    inter = max(0, (r - l))
    union = area1 + area2 - inter
    iou = inter / union
    return iou

def ffmpeg_once(src_path: str, dst_path: str, *, fps: int = None, resolution: int = None, pad: str = '#000000', mode='bicubic'):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    command = [
        './ffmpeg/ffmpeg',
        # '-y',
        '-n',       # skip if target file exists
        '-sws_flags', mode,
        '-i', src_path,
        '-an',
        '-threads', '10',
    ]
    if fps is not None:
        command += ['-r', str(fps)]
    if resolution is not None:
        command += ['-vf', f"scale='if(gt(iw\\,ih)\\,{resolution}\\,-2)':'if(gt(iw\\,ih)\\,-2\\,{resolution})',pad={resolution}:{resolution}:(ow-iw)/2:(oh-ih)/2:color='{pad}'"]
    command += [dst_path]
    subprocess.run(command, check=True)

def distributed_ffmpeg(*, src_root: str, fps: int = None, resolution: int = None, pad: str = '#000000', mode='bicubic'):
    import submitit
    env = submitit.JobEnvironment()
    src_root = src_root.rstrip('/')
    pather = pathlib.Path(src_root)
    src_paths = [str(path) for path in pather.rglob('*') if path.is_file() and str(path).endswith('.mp4')]
    dst_root = src_root
    if fps is not None:
        dst_root += f'_{fps}fps'
    if resolution is not None:
        assert (pad is not None)
        dst_root += f'_max{resolution}'
    for i, src_path in tqdm.tqdm(enumerate(src_paths), desc=f'{src_root} -> {dst_root}'):
        if i % env.num_tasks != env.global_rank:
            continue
        dst_path = src_path.replace(src_root, dst_root)
        ffmpeg_once(src_path, dst_path, fps=fps, resolution=resolution, pad=pad, mode=mode)

def distributed_encode(*, src_root: str, vision_pretrained: str, vision_encode: callable, batch_size: int, embed_mark: str, save_bf16: bool = False, **kwargs):
    env = submitit.JobEnvironment()
    src_root = src_root.rstrip('/')
    model = AutoModel.from_pretrained(vision_pretrained, device_map=f'cuda:{env.local_rank}').vision_model
    model.eval()
    dst_root = f"{src_root}_{embed_mark.split('_')[-1]}_{vision_pretrained.replace('/', '--')}"
    os.makedirs(dst_root, exist_ok=True)
    for i, file in tqdm.tqdm(enumerate(os.listdir(src_root)), desc=f'{src_root} -> {dst_root}'):
        if i % env.num_tasks != env.global_rank:
            continue
        frame_path = os.path.join(src_root, file)
        save_path = os.path.splitext(frame_path)[0] + '.pt'
        save_path = save_path.replace(src_root, dst_root)
        frames = torchvision.io.read_video(frame_path, pts_unit='sec', output_format='TCHW')[0]
        with torch.no_grad():
            frames = torch.cat([vision_encode(model, batch.to(f'cuda:{env.local_rank}')).cpu() for batch in frames.split(batch_size)])
        if save_bf16:
            frames = frames.to(torch.bfloat16)
        torch.save(frames, save_path)

def round_time_by_fps(time: float, fps: int, min_time: float, max_time: float):
    return min(max(round(time * fps) / fps, min_time), max_time)

def ceil_time_by_fps(time: float, fps: int, min_time: float, max_time: float):
    return min(max(math.ceil(time * fps) / fps, min_time), max_time)

def floor_time_by_fps(time: float, fps: int, min_time: float, max_time: float):
    return min(max(math.floor(time * fps) / fps, min_time), max_time)

def resize_and_pad_frame(frame, output_size, pad_color=(0, 0, 0)):
    input_height, input_width = frame.shape[:2]
    if input_height == output_size and input_width == output_size:
        return frame
    if input_width > input_height:
        # Landscape video: scale width to the resolution, adjust height
        new_width = output_size
        new_height = int((input_height / input_width) * output_size)
    else:
        # Portrait video: scale height to the resolution, adjust width
        new_height = output_size
        new_width = int((input_width / input_height) * output_size)
    resized_frame = cv2.resize(frame, (new_width, new_height))
    # pad the frame
    canvas = cv2.copyMakeBorder(
        resized_frame,
        top=(output_size - new_height) // 2,
        bottom=(output_size - new_height + 1) // 2,
        left=(output_size - new_width) // 2,
        right=(output_size - new_width + 1) // 2,
        borderType=cv2.BORDER_CONSTANT,
        value=pad_color
    )
    return canvas
