from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
import json

app = Flask(__name__)

# Load model and data
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

with open('campus_buildings.json') as f:
    data = json.load(f)

# Extract searchable entries (building_key, building_name, room_id, room_name)
search_entries = []

for building in data['buildings']:
    search_entries.append({
        "type": "building",
        "key": building['building_key'],
        "name": building['building_name'],
        "description": building['building_description'],
        "image": building['building_image'],
        "coordinates": building['coordinates']
    })
    for floor in building['floors']:
        for room in floor['rooms']:
            search_entries.append({
                "type": "room",
                "id": room.get('room_id', ''),
                "name": room['room_name'],
                "location": f"{building['building_name']} | {floor['floor_number']} floor",
                "image": building['building_image'],
                "room_type": room.get('room_type', ''),
                "coordinates": building['coordinates']
            })

# Precompute embeddings
texts = [
    f"{entry.get('key', '')} {entry['name']}" if entry['type'] == "building"
    else f"{entry.get('id', '')} {entry['name']}"
    for entry in search_entries
]
embeddings = model.encode(texts, convert_to_tensor=True)

@app.route('/search', methods=['POST'])
def search():
    query = request.json['query']
    query_embedding = model.encode(query, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]

    # Top 10 matches
    top_results = scores.topk(10)
    results = [search_entries[i] for i in top_results[1]]

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
