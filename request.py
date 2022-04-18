import pip._vendor.requests

url = 'http://localhost:5000/'
r = pip._vendor.requests.post(url,json={'experience':2, 'test_score':9, 'interview_score':6})

print(r.json())