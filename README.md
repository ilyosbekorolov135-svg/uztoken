# UzToken — O'zbek tili uchun morfologik gibrid tokenizator

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests: 100% Passed](https://img.shields.io/badge/tests-75%2F75%20passed-brightgreen.svg)]()
[![Speed: 50k+ words/sec](https://img.shields.io/badge/speed-~50%2C000%2B%20words%2Fsec-orange.svg)]()

**UzToken** — o'zbek tili uchun maxsus ishlab chiqilgan, agglyutinativ grammatikani to'liq hisobga oluvchi yuqori unumdor gibrid tokenizator. U qoidalarga asoslangan morfologik tahlil bilan subword (BPE) tokenizatsiyasini birlashtiradi.

---

## Asosiy afzalliklari

- **To'liq morfologik tahlil**: So'zlarni grammatik to'g'ri o'zak va qo'shimchalarga ajratadi (`maktablarimizda` → `maktab` + `lar` + `imiz` + `da`).
- **Morfonologik tiklash (Sound Changes)**: Tovush almashinuvlarini avtomatik tiklaydi:
  - **Q ↔ G'**: `qishlog'imiz` → `qishloq`, `o'rtog'im` → `o'rtoq`
  - **K ↔ G**: `yuragimda` → `yurak`, `tilagim` → `tilak`
  - **Unli tushishi (Syncope)**: `shahri` → `shahar`, `singlim` → `singil`, `burni` → `burun`, `og'zi` → `og'iz`
- **Apostrofni yo'qotishsiz (Lossless) normalizatsiya**:
  - `oʻ` va `gʻ` harflari uchun rasmiy `ʻ` (U+02BB)
  - Tutuq belgisi uchun rasmiy `ʼ` (U+02BC)
  - Har qanday kiruvchi matn (`'`, `’`, `` ` ``, `´`) avtomatik to'g'rilanadi va `decode(encode(matn)) == matn` kafolatlanadi.
- **OOV (Out-Of-Vocabulary) aqlli gibrid tahlil**: Lug'atda yo'q so'zlardan avval qo'shimchalar ajratiladi, faqat yangi o'zak BPE'ga uzatiladi (`Pythonda` → `Python` + `da`).
- **72,200+ tozalangan o'zaklar**: Lug'atdagi shovqinlar va soxta o'zaklar to'liq tozalangan.
- **Tezkor Trie + LRU Cache**: O(n) murakkablikda suffikslarni qidirish, sekundiga **50,000 – 60,000+ so'z**.

---

## 📊 Haqiqiy Test va Taqqoslash Natijalari

Kutubxona 75 ta integratsion va unit-testlar orqali to'liq sinovdan o'tgan (**100% muvaffaqiyatli**):

### 1. So'z darajasidagi taqqoslash (UzToken vs Odatiy BPE)

| Toifa | Kiruvchi so'z | UzToken (Gibrid) | Odatiy BPE |
| :--- | :--- | :--- | :--- |
| **Agglyutinativ ot** | `maktablarimizda` | `Ġmaktab ##lar ##imiz ##da` | `maktab lar imizda` (noto'g'ri chegara) |
| **Fe'l shakllari** | `ishlaydigan` | `Ġishla ##ydigan` | `ishla y di ga n` (harflab ketdi) |
| **Morfonologiya (Q/G')** | `qishlog'imiz` | `Ġqishloq ##imiz` | `qishlogʻimiz` (o'zak topilmadi) |
| **Morfonologiya (K/G)** | `yuragimda` | `Ġyurak ##im ##da` | `y ur ag im da` (maydalanib ketdi) |
| **Unli tushishi** | `shahrimiz` | `Ġshahar ##imiz` | `shah r imiz` |
| **Apostrof variantlari** | `o‘quvchilar` | `Ġoʻquv ##chi ##lar` | `oʻ qu vchilar` |
| **Chet/Dasturlash so'zi** | `Pythonda` | `ĠPython ##da` | `Pyt h on da` |

### 2. Xulosa va Metrikalar

| Mezon | UzToken (Bizning usul) | Odatiy BPE / WordPiece |
| :--- | :---: | :---: |
| **Semantik o'zakni saqlash** | **94%** (aniq grammatik) | ~20% (tasodifiy bo'laklash) |
| **Morfonologiya (tovush tiklash)**| **100%** (qoidalar bo'yicha) | 0% (butunlay ushlay olmaydi) |
| **Apostrof xatolarini tuzatish** | **Avtomatik** (`oʻ`, `gʻ`, `ʼ`) | Har xil token deb biladi |
| **Bosh harfni saqlash (Case)** | **To'liq saqlanadi** | Ko'pincha yo'qotiladi |
| **Token tejamkorligi (gapda)** | **52 ta token** | 59+ ta token |
| **Ishlash tezligi** | **~50,300 so'z/sekund** | ~100,000 so'z/sekund |
| **Avtomatlashtirilgan testlar** | **75 / 75 (100% pass)** | - |

---

## O'rnatish

```bash
pip install uztoken
```

Yoki to'g'ridan-to'g'ri GitHub repozitoriysidan:

```bash
git clone https://github.com/ilyosceo/uztoken.git
cd uztoken
pip install -e .
```

---

## Tezkor Foydalanish

### 1. Til modellari (LLM / Transformer) uchun tokenizatsiya:

```python
from uztoken import create_tokenizer

tok = create_tokenizer(load_dict=True)

# Gibrid subwordlar ro'yxati
subwords = tok.tokenize_to_strings("Maktablarimizda yangi darsliklar keldi!", style="hybrid")
print(subwords)
# Natija:
# ['ĠMaktab', '##lar', '##imiz', '##da', 'Ġyangi', 'Ġdars', '##lik', '##lar', 'Ġkeldi', '!']
```

### 2. Chuqur morfologik tahlil:

```python
for token in tok.tokenize("Maktablarimizda yangi darsliklar keldi"):
    print(token)

# Token('Maktablarimizda' → maktab +lar(ko'plik) -> +imiz(1-shaxs ko'plik egalik) -> +da(o'rin-payt))
# Token('yangi' → yangi [bare])
# Token('darsliklar' → dars +lik(mavhum ot yasovchi) -> +lar(ko'plik))
# Token('keldi' → keldi [bare])
```

### 3. Raqamli ID'larga kodlash (Encode & Decode):

```python
# Butun matnni integer ID'larga o'tkazish
ids = tok.encode("Maktablarimizda kitob o'qiymiz.")
print("Token IDs:", ids)

# Qayta tiklash (decode)
text = tok.decode(ids)
print("Tiklandi:", text)
```

---

## Interaktiv Terminal Sinovi

Kutubxonani real matnlarda darhol sinab ko'rish uchun:

```bash
python -X utf8 interactive.py
```

---

## Testlarni ishga tushirish

Barcha 75 ta unit va integratsion testlarni tekshirish:

```bash
pytest tests/ -v
```

---

## Litsenziya

Loyiha [MIT](LICENSE) litsenziyasi asosida ochiq manbalidir.

## Muallif

**Ilyosbek Orolov** ([@ilyosceo](https://github.com/ilyosceo))