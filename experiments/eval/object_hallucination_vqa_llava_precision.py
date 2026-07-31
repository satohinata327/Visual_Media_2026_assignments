import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import math

# import kornia
from transformers import set_seed
from vcd_utils.vcd_add_noise import add_diffusion_noise
# 公式VCD samplingではなく、No高確信度時にclean logitsを維持する
# Precision-Gated VCD samplingを読み込む。
from vcd_utils.vcd_sample_precision import evolve_vcd_sampling
evolve_vcd_sampling()


def get_single_token_id(tokenizer, text):
    # Gateで参照するYes/Noが単一tokenであることを明示的に確認する。
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(
            f"{text!r}が単一tokenではありません: token_ids={token_ids}"
        )
    return token_ids[0]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # 独自引数をmodel.generateへ直接渡すとTransformersの引数検証に
    # 抵触するため、samplingから参照するmodel属性として保持する。
    if args.precision_gate_threshold is not None:
        if not args.use_cd:
            raise ValueError(
                "--precision_gate_thresholdは--use_cdと一緒に指定してください。"
            )
        if not 0.0 <= args.precision_gate_threshold <= 1.0:
            raise ValueError(
                "--precision_gate_thresholdは0以上1以下で指定してください。"
            )

        model.precision_gate_yes_token_id = get_single_token_id(
            tokenizer,
            "Yes",
        )
        model.precision_gate_no_token_id = get_single_token_id(
            tokenizer,
            "No",
        )

    model.precision_gate_threshold = args.precision_gate_threshold

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs + " Please answer this question with one word.")
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image = Image.open(os.path.join(args.image_folder, image_file))
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        
        if args.use_cd:
            image_tensor_cd = add_diffusion_noise(image_tensor, args.noise_step)
        else:
            image_tensor_cd = None      

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        # 前の質問のGate情報が残らないよう、各推論前に初期化する。
        model.precision_gate_triggered = False
        model.precision_gate_p_no = None

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                images_cd=(image_tensor_cd.unsqueeze(0).half().cuda() if image_tensor_cd is not None else None),
                cd_alpha = args.cd_alpha,
                cd_beta = args.cd_beta,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=1024,
                use_cache=True)

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        # 改良版と公式VCDを後から区別し、Gateの発動条件を分析できるよう
        # decoding条件とclean側のNo確信度を各行へ保存する。
        if args.precision_gate_threshold is not None:
            decoding = "precision_gated_vcd"
        elif args.use_cd:
            decoding = "vcd"
        else:
            decoding = "regular"

        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "model_id": model_name,
                                   "image": image_file,
                                   "metadata": {
                                       "decoding": decoding,
                                       "seed": args.seed,
                                       "noise_step": args.noise_step if args.use_cd else None,
                                       "cd_alpha": args.cd_alpha if args.use_cd else None,
                                       "cd_beta": args.cd_beta if args.use_cd else None,
                                       "precision_gate_threshold": args.precision_gate_threshold,
                                       "precision_gate_triggered": model.precision_gate_triggered,
                                       "clean_p_no": model.precision_gate_p_no,
                                   }}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)

    parser.add_argument("--noise_step", type=int, default=500)
    parser.add_argument("--use_cd", action='store_true', default=False)
    parser.add_argument("--cd_alpha", type=float, default=1)
    parser.add_argument("--cd_beta", type=float, default=0.1)
    # Noneなら公式VCD、数値を指定するとPrecision-Gated VCDを実行する。
    parser.add_argument(
        "--precision_gate_threshold",
        "--precision-gate-threshold",
        dest="precision_gate_threshold",
        type=float,
        default=None,
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args)
