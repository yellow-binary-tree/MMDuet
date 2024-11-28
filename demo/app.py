import os, torchvision, transformers, time
import gradio as gr
from threading import Event

from models import parse_args
from demo.liveinfer import LiveInferForDemo, load_video
logger = transformers.logging.get_logger('liveinfer')

args = parse_args('test')
args.stream_end_prob_threshold = 0.3
liveinfer = LiveInferForDemo(args)

pause_event = Event()  # Event for pausing/resuming
pause_event.set()      # Initially, processing is allowed (not paused)

css = """
    #gr_title {text-align: center;}
    #gr_video {max-height: 480px;}
    #gr_chatbot {max-height: 480px;}
"""


class HistorySynchronizer:
    def __init__(self):
        self.history = []

    def set_history(self, history):
        self.history = history

    def get_history(self):
        return self.history

    def reset(self):
        self.history = []

history_synchronizer = HistorySynchronizer()


class ChatInterfaceWithUserMsgTime(gr.ChatInterface):
    async def _display_input(
        self, message: str, history
    ):
        message = f"[time={liveinfer.video_time:.1f}s] {message}"
        history = history_synchronizer.get_history()
        if isinstance(message, str) and self.type == "tuples":
            history.append([message, None])  # type: ignore
        elif isinstance(message, str) and self.type == "messages":
            history.append({"role": "user", "content": message})  # type: ignore
        history_synchronizer.set_history(history)
        return history  # type: ignore


with gr.Blocks(title="MMDuet", css=css) as demo:
    gr.Markdown("# VideoLLM Knows When to Speak: Enhancing Time-Sensitive Video Comprehension with Video-Text Duet Interaction Format", elem_id='gr_title')
    with gr.Row():      # row for instructions
        with gr.Column():
            gr.Markdown((
                'This demo demonstrates **MMDuet**, a VideoLLM you can interact with in a real-time manner while the video plays.\n'
                '## Usage\n'
                '1. Upload the video, and set the "Threshold Mode", "Scores Used", "Remove Previous Model Turns in Context" and "Threshold" hyperparameters. After this click "Start Chat", the frames will be sampled from the video and encoded by the ViT of MMDuet,'
                'then the video will start to progress in the bottom left corner.\n'
                '1. When the video is progressing, if you want to send a message, type in the message box and click "Submit". Your message will be inserted at the current position of the video.\n'
                'You can also pause the video before typing & submitting your message.\n'
                '1. The conversation will be listed in the chatbot in the bottom right corner.\n'
                '1. If you want to change the video or hyperparameters, click the red "Stop Video" button.\n'
                '## Hyparameters\n'
                '1. **Threshold Mode**: When "single-frame score" is selected, MMDuet responses when the video score of current frame exceeds the "Score Threshold".'
                'When "sum score" is selected, MMDuet responses when the sum of video scores of the several latest frames exceeds the "Score Threshold".\n'
                '1. **Scores Used**: The video score is set to the sum of the scores selected here.\n'
                '1. **Remove Previous Model Turns in Context**: Whether to remove the previous model-generated responses from the dialogue context.'
                'If set to "yes", The model will not be affected by previous content when generating subsequent responses, which will significantly reduce the occurrence of duplicate responses.'
                'However, this is at the cost of losing those context information.\n'
                '1. **Threshold**: The threshold for the video score to trigger a response from MMDuet.\n'
                '1. **Frame Interval**: The time interval between frames sampled from the video.\n'
                '## Examples\n'
                'In the upper right part we list 3 examples. The first 2 examples are about answering questions about videos in a real-time manner, and the third question is about dense video captioning for a **10-minutes long** video.\n'
                'Choose the example, click "Start Chat", and you will see an example user message filled in the message input box. Click "Submit" to submit it as early in the video as possible.\n'
                '## Notes\n'
                '- For a clear demonstration, we add a short `time.sleep()` after each frame. Therefore, the video progresses much slower than the actual inference speed of MMDuet.\n'
                '- If the app stuck for some reason, refreshing the page usually doesn\'t help. Restart the app and try again.'
            ))

    with gr.Row(visible=False):
        def handle_user_input(message, history):
            liveinfer.encode_given_query(message)
        gr_chat_interface = ChatInterfaceWithUserMsgTime(
            fn=handle_user_input,
            chatbot=gr.Chatbot(
                elem_id="gr_chatbot",
                label='chatbot',
                avatar_images=('demo/assets/user_avatar.png', 'demo/assets/assistant_avatar.png'),
                render=False
            ),
            retry_btn=None, undo_btn=None,
            examples=[],
        )

    with gr.Row(), gr.Blocks() as hyperparam_block:
        gr_video = gr.Video(label="Input Video", visible=True, sources=['upload'], autoplay=False)

        with gr.Column():
            # choose the threshold before starting inference
            gr_thres_mode = gr.Radio(choices=["single-frame score", "sum score"], value="single-frame score", label="Threshold Mode")
            gr_used_scores = gr.CheckboxGroup(choices=["informative score", "relevance score"], value=["informative score"], label="Scores Used")
            gr_rm_ass_turns = gr.Radio(choices=["yes", "no"], value="yes", label="Remove Previous Model Turns in Context")
            gr_threshold = gr.Slider(minimum=0, maximum=3, step=0.05, value=args.stream_end_prob_threshold, interactive=True, label="Score Threshold")
            gr_frame_interval = gr.Slider(minimum=0.1, maximum=10, step=0.1, value=(1/args.frame_fps), interactive=True, label="Frame Interval (sec)")
            gr_start_button = gr.Button("Start Chat", variant="primary")

        gr_examples = gr.Examples(
            examples=[
                ["demo/assets/office.mp4", "single-frame score", ["informative score", "relevance score"], 0.4, 0.5, "What is happening in the office?", "no"],
                ["demo/assets/drive.mp4", "single-frame score", ["informative score", "relevance score"], 0.4, 0.5, "Who is driving the car?", "no"],
                ["demo/assets/cooking.mp4", "sum score", ["informative score"], 1.5, 2.0, "Please simply describe what do you see.", "yes"],
            ],
            inputs=[gr_video, gr_thres_mode, gr_used_scores, gr_threshold, gr_frame_interval, gr_chat_interface.textbox, gr_rm_ass_turns],
            # outputs=[gr_video, gr_thres_mode, gr_used_scores, gr_threshold, gr_frame_interval, gr_chat_interface.examples],
            label="Examples"
        )

    with gr.Row() as chat:
        with gr.Column():
            gr_frame_display = gr.Image(label="Current Model Input Frame", interactive=False)
            with gr.Row():
                gr_time_display = gr.Number(label="Current Video Time", value=0)
            with gr.Row():
                gr_inf_score_display = gr.Number(label="Informative Score", value=0)
                gr_rel_score_display = gr.Number(label="Relevance Score", value=0)
            with gr.Row():
                gr_pause_button = gr.Button("Pause Video")
                gr_stop_button = gr.Button("Stop Video", variant='stop')

        with gr.Column():
            gr_chat_interface.render()

    def start_chat(src_video_path, thres_mode, rm_ass_turns, scores, threshold, frame_interval, history):
        # clear the chatbot and frame display
        yield 0, 0, 0, None, []

        # set the hyperparams
        liveinfer.reset()
        history_synchronizer.reset()
        liveinfer.score_heads = [s.replace(' ', '_') for s in scores]
        if thres_mode == 'single-frame score':
            liveinfer.stream_end_prob_threshold = threshold
            liveinfer.stream_end_score_sum_threshold = None
        elif thres_mode == 'sum score':
            liveinfer.stream_end_prob_threshold = None
            liveinfer.stream_end_score_sum_threshold = threshold
        liveinfer.remove_assistant_turns = rm_ass_turns == 'yes'

        # disable the hyperparams
        for component in hyperparam_block.children:
            component.interactive = False

        # upload the video
        frame_fps = 1 / frame_interval
        liveinfer.set_fps(frame_fps)
        video_input, original_frame_list = load_video(src_video_path, frame_fps)
        liveinfer.input_video_stream(video_input)

        while liveinfer.frame_embeds_queue:
            start_time = time.time()
            pause_event.wait()
            ret = liveinfer.input_one_frame()
            history = history_synchronizer.get_history()

            if ret['response'] is not None:
                history.append((None, f"[time={ret['time']}s] {ret['response']}"))
                history_synchronizer.set_history(history)

            elapsed_time = time.time() - start_time
            target_delay_time = min(frame_interval, 0.2)
            if elapsed_time < target_delay_time:       # or the video plays too fast
                time.sleep(frame_interval - elapsed_time)
            yield ret['time'], ret['informative_score'], ret['relevance_score'], \
                original_frame_list[ret['frame_idx'] - 1], history

    gr_start_button.click(
        fn=start_chat,
        inputs=[gr_video, gr_thres_mode, gr_rm_ass_turns, gr_used_scores, gr_threshold, gr_frame_interval, gr_chat_interface.chatbot],
        outputs=[gr_time_display, gr_inf_score_display, gr_rel_score_display, gr_frame_display, gr_chat_interface.chatbot]
    )

    def toggle_pause():
        if pause_event.is_set():
            pause_event.clear()  # Pause processing
            return "Resume Video"
        else:
            pause_event.set()    # Resume processing
            return "Pause Video"

    gr_pause_button.click(
        toggle_pause,
        inputs=[],
        outputs=gr_pause_button
    )

    def stop_chat():
        liveinfer.reset()
        history_synchronizer.reset()

        # enable the hyperparams
        for component in hyperparam_block.children:
            component.interactive = True

        return 0, 0, 0, None, []

    gr_stop_button.click(
        stop_chat,
        inputs=[],
        outputs=[gr_time_display, gr_inf_score_display, gr_rel_score_display, gr_frame_display, gr_chat_interface.chatbot]
    )

    demo.queue()
    demo.launch(share=False)
