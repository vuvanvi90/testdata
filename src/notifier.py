import requests
from datetime import datetime

# ==========================================
# CẤU HÌNH TELEGRAM CỦA BẠN
# ==========================================
TELEGRAM_TOKEN = "8304516858:AAEPmQmNr2wrI31yvszBNjOJ5HOjunTG79k"
TELEGRAM_CHAT_ID = "-5129050658"

def send_telegram_alert(message):
    """Hàm gửi tin nhắn qua Telegram Bot"""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
        
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    
    # Gắn thêm Header thời gian cho chuyên nghiệp
    current_time = datetime.now().strftime('%d/%m/%Y %H:%M')
    # formatted_msg = f"🤖 <b>QUANT SYSTEM ALERT</b>\n🕒 {current_time}\n{'-'*30}\n{message}"
    formatted_msg = f"{message}"
    
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": formatted_msg,
        "parse_mode": "HTML" # Cho phép dùng thẻ <b>, <i> để làm đậm chữ
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            print(f"\n[🚀] Đã bắn tín hiệu Telegram thành công!")
        else:
            print(f"\n[!] Lỗi gửi Telegram: {response.text}")
    except Exception as e:
        print(f"\n[!] Không thể kết nối tới Telegram: {e}")

# hàm này để lấy id của group chat
def send_telegram_updates(message):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
        response = requests.get(url)
        data = response.json()
        for update in data["result"]:
            print(update)
    except Exception as e:
        print(f"\n[!] Không thể kết nối tới Telegram: {e}")

# Chạy test thử luôn khi vừa lưu file
if __name__ == "__main__":
    send_telegram_alert("✅ Hệ thống Telegram đã kết nối thành công!")