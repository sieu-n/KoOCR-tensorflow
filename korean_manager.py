import json 
import numpy as np

CHOSUNG_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ', ' ']
# 중성 리스트. 00 ~ 20
JUNGSUNG_LIST = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ', ' ']
# 종성 리스트. 00 ~ 27 + 1(1개 없음)
JONGSUNG_LIST = [' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
    
def load_charset(charset='kr',charset_path='files/cjk.json'):
  #Load charset(list of characters): charset[index] = korean
  #Charset referenced from zi2zi: kr, jp, gbk2312, gbk2312_t, gbk

  with open(charset_path) as json_file:
    data = json.load(json_file)
    charset=data[charset]
  return charset
def inverse_charset(charset='kr',charset_path='files/cjk.json'):
  #Load the inverse of the charset: inverse[korean] = index
  charset=load_charset(charset=charset,charset_path=charset_path)
  inverse={}
  for x in range(len(charset)):
    inverse[charset[x]]=x
  return inverse

def index_to_korean(l):
  cho,jung,jong=l
  characterValue = ( (cho * 21) + jung) * 28 + jong + 0xAC00
  return chr(characterValue)

def korean_numpy(words):
  charset=inverse_charset()
  arr=[]
  for w in words:
    arr.append(charset[w])
  return arr
def korean_split_numpy(words,to_text=False):
  # 한글 글자의 np array를 입력받아 초성, 중성, 종성을 각각의 array로 내보내는 함수
  cho,jung,jong=[],[],[]
  for w in words:
    ch1 = (ord(w) - ord('가'))//588
    ch2 = ((ord(w) - ord('가')) - (588*ch1)) // 28
    ch3 = (ord(w) - ord('가')) - (588*ch1) - 28*ch2
    
    #ㄱ,ㄴ,ㄷ,... 처리 
    if ch1==-54:
      ch1,ch2,ch3=19,22,ord(w) - ord('ㄱ')+1

    if to_text:
      cho.append(CHOSUNG_LIST[ch1])
      jung.append(JUNGSUNG_LIST[ch2])
      jong.append(JONGSUNG_LIST[ch3])
    else:
      cho.append(ch1)
      jung.append(ch2)
      jong.append(ch3)

  return np.array(cho),np.array(jung),np.array(jong)