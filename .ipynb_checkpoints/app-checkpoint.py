import os
import time
import logging
import subprocess
import tempfile
import whisper
import requests
from datetime import datetime
from pathlib import Path
from flask import Flask, request, abort, jsonify, send_file
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, AudioMessage, TextMessage, TextSendMessage, AudioSendMessage
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import opencc
app = Flask(__name__)
app.secret_key = os.urandom(24)
logging.basicConfig(level=logging.INFO)
# LINE bot 設置
LINE_CHANNEL_ACCESS_TOKEN = 'OG/Z9nCIEAvPSMNHsdfcOXS3ulWyNQihSszijTJdXEfa7srZYeuTKpUkarrwf6TXcEOwjp/6EH+0l/aceCyrQWU9KUnRlFvdl47EFT8P+0GiMvKwa03TJbAy9c7sUk/Z8l50wjUNz5n/LOXtDAcK/QdB04t89/1O/w1cDnyilFU='
LINE_CHANNEL_SECRET = 'f64b31faefee2d64598ff9c317f8ce43'
SERVER_URL = 'https://linebot-smq6.onrender.com'
STT_API_URL = 'http://180.218.16.187:30303/recognition_long_audio'
TTS_API_URL = 'http://180.218.16.187:30303/getTTSfromText'
LLM_API_URL = 'http://61.66.218.215:30315/llm_chat'
SERVER_PORT = 8080
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
line_handler = WebhookHandler(LINE_CHANNEL_SECRET)
# Whisper 和 LLM 模型設置
stt_model = whisper.load_model("tiny")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
llm_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct").to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
converter = opencc.OpenCC('s2t')
# 模型功能
def get_text_from_audio(audio_path):
    with open(audio_path, 'rb') as f:
        files = {'audio': (os.path.basename(audio_path), f, 'audio/mpeg')}
        response = requests.post(STT_API_URL, files=files)
        return response.json().get('result', '無法辨識音訊') if response.status_code == 200 else '錄音語音品質不佳，請再試試。'
def get_response_from_llm(query):
    payload = {'token': 'TEST', 'query': query, 'prompt_name': '艾妮機器人', 'max_tokens': '1024'}
    response = requests.post(LLM_API_URL, data=payload)
    return response.json().get('result', '無法獲取回應')
def get_audio_from_text(text):
    payload = {'tone': '0', 'speed': '0', 'content': text, 'gender': '1'}
    response = requests.post(TTS_API_URL, data=payload)
    audio_path = f'static/{int(time.time())}.mp3'
    with open(audio_path, 'wb') as f:
        f.write(response.content)
    return audio_path
def run_command(command):
    try:
        result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8", timeout=3600)
        return result.returncode == 0, result.stdout if result.returncode == 0 else result.stderr
    except Exception as e:
        return False, str(e)
# 路由和處理函數
@app.route("/", methods=["GET"])
def home():
    return "Line Bot 已啟動並運作"
@app.route("/webhook", methods=["POST"])
def callback():
    signature = request.headers["X-Line-Signature"]
    body = request.get_data(as_text=True)
    try:
        line_handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return "OK"
@app.route("/qa", methods=["POST"])
def qa():
    question = request.json.get('text')
    if not question:
        return jsonify({"error": "未提供文本"}), 400
    answer = answer_question(question)
    return jsonify({"answer": answer})
@app.route("/transcribe", methods=["POST"])
def transcribe():
    audio_file = request.files.get('file')
    if not audio_file:
        return jsonify({"error": "未上傳音訊檔案"}), 400
    with tempfile.NamedTemporaryFile(delete=False) as temp_audio_file:
        audio_file.save(temp_audio_file.name)
        audio = whisper.load_audio(temp_audio_file.name)
        result = stt_model.transcribe(audio, language='zh')
    os.remove(temp_audio_file.name)
    return jsonify({"transcription": result['text']})
@app.route('/synthesize', methods=['POST'])
def synthesize():
    content = request.form.get('content', '上傳資料內容有誤')
    gender = request.form.get('gender', '1')
    tone = request.form.get('tone', '0')
    speed = request.form.get('speed', '0')
    voices = {'1': "zh-TW-YunJheNeural", '0': "zh-TW-HsiaoYuNeural"}
    voice = voices.get(gender, "zh-TW-HsiaoChenNeural")
    pitch = f"{tone}Hz"
    rate = f"{speed}%"
    
    output_path = f"static/{int(time.time() * 1000)}.mp3"
    command = f'edge-tts --text "{content}" --write-media "{output_path}" --voice "{voice}" --pitch="{pitch}" --rate="{rate}"'
    success, message = run_command(command)
    if success:
        return send_file(output_path, mimetype='audio/mpeg')
    return jsonify({"error": message}), 500
# 問答功能
def answer_question(question):
    inputs = tokenizer(question, return_tensors="pt", padding=True).to(llm_model.device)
    outputs = llm_model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], max_new_tokens=256, pad_token_id=tokenizer.eos_token_id)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return converter.convert(answer)
# Line handlers
@line_handler.add(MessageEvent, message=AudioMessage)
def handle_audio_message(event):
    audio_path = f'static/{int(time.time())}.mp3'
    with open(audio_path, 'wb') as fd:
        for chunk in line_bot_api.get_message_content(event.message.id).iter_content():
            fd.write(chunk)
    text = get_text_from_audio(audio_path)
    llm_response = get_response_from_llm(text)
    reply_audio_path = get_audio_from_text(llm_response)
    if os.path.exists(reply_audio_path):
        line_bot_api.reply_message(
            event.reply_token,
            [
                TextSendMessage(text=llm_response),
                AudioSendMessage(original_content_url=f'{SERVER_URL}/{reply_audio_path}', duration=330)
            ]
        )
    else:
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text="合成語音時錯誤，請檢查 TTS Server"))
@line_handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    text = event.message.text
    llm_response = get_response_from_llm(text)
    reply_audio_path = get_audio_from_text(llm_response)
    if os.path.exists(reply_audio_path):
        line_bot_api.reply_message(
            event.reply_token,
            [
                TextSendMessage(text=llm_response),
                AudioSendMessage(original_content_url=f'{SERVER_URL}/{reply_audio_path}', duration=330)
            ]
        )
    else:
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text="合成語音時錯誤，請檢查 TTS Server"))
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=SERVER_PORT, use_reloader=False)