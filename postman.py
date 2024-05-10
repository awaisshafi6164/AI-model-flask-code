import requests

url = "http://192.168.0.113:5001/summarize-text"
data = {"text": "Twitter is a popular social network which allows millions of users to share their opinions on what happens all over the world . In this work we present a system for real time Twitter data analysis in order to follow popular events from the user s perspective . The method we propose extends and improves the Soft Frequent Pattern Mining algorithm by overcoming its limitations in dealing with dynamic real time detection scenarios . In particular in order to obtain timely results the stream of tweets is organized in dynamic windows whose size depends both on the volume of tweets and time . Since we aim to highlight the user s point of view the set of keywords used to query Twitter is progressively refined to include new relevant terms which reflect the emergence of new subtopics or new trends in the main topic . The real time detection system has been evaluated during the 2014 FIFA World Cup and experimental results show the effectiveness of our solution"}
response = requests.post(url, json=data)

if response.status_code == 200:
    result = response.json()
    print("summarize-text:", result['summary'])
else:
    print("Error:", response.text)
