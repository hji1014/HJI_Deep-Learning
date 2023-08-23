# 내가 설정한 디스코드 서버로 메세지 전송
# '딥러닝 학습 완료 알림' 등의 용도로 사용할 수 있음

#########
## 예시 ##
#########

avg_pt = 100

def DL_complete():
    from knockknock import discord_sender
    webhook_url = "https://discord.com/api/webhooks/1014081244734173274/kCGlk4rXRPb4LSOdf4ECz9P0lvbiHr5tq4EOR2Zf6RPd8OgU8eytgtG2-IjER1abFt4y"
    @discord_sender(webhook_url=webhook_url)
    def DL_notification():
        #import time
        #time.sleep(3)
        #return {'averaged test acc of premodel' : avg_pt}, {'averaged test acc of transfer model' : avg_tl}, {'소요시간' :(terminate_time - start_time)} # Optional return value
        return {'averaged test acc of premodel': avg_pt}
    DL_notification()

DL_complete()