import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
import cupy as cp
import time


class DataAnalysis:
    def __init__(self,referenceStructure):
        self.referenceStructure=referenceStructure
        self.analyzer=SentenceTransformer("all-MiniLM-L6-v2").to("cuda").half()
        self.encodedReferences=[]
        for key in self.referenceStructure.keys():
            subSet=[]
            for item in self.referenceStructure[key]:
                subSet.append(self.analyzer.encode(item, convert_to_tensor=True, device="cuda").half())
            self.encodedReferences.append(torch.stack(subSet).mean(dim=0))
        self.encodedReferences=torch.stack(self.encodedReferences).to("cuda").half()
    def getEncoding(self, data):
        start=time.time()
        inputEncoded=self.analyzer.encode(data, convert_to_tensor=True, device="cuda").half()
        end=time.time()
        encoding=f.cosine_similarity(
            torch.tensor(inputEncoded).unsqueeze(1), 
            self.encodedReferences.unsqueeze(0),
            dim=2
        )
        return encoding
    def getStructure(self):
        return self.encodingStructure

'''test={
    "programming": [
        "How do I implement a binary search algorithm?",
        "What's the difference between stack and heap memory?",
        "Explain object-oriented programming principles"
    ],
    
    "python": [
        "How do I use list comprehensions in Python?",
        "What's the difference between lists and tuples in Python?",
        "How to handle file I/O in Python?"
    ],
    
    "literature": [
        "Analyze the themes in Shakespeare's Hamlet",
        "What are the characteristics of Romantic poetry?",
        "Explain the symbolism in The Great Gatsby"
    ],
    
    "conversational": [
        "How are you doing today?",
        "What's your favorite movie?",
        "Tell me about your weekend plans"
    ],
    
    "machine_learning": [
        "How do neural networks work?",
        "What's the difference between supervised and unsupervised learning?",
        "Explain gradient descent algorithm"
    ]
}

data=[
    # Programming (General)
    "How do I implement a binary search algorithm?",
    "What's the difference between stack and heap memory?",
    "Explain object-oriented programming principles",
    "How to optimize database queries for better performance?",
    "What are design patterns and when should I use them?",
    "How do I handle exceptions in my code?",
    "What's the best way to structure a large codebase?",
    "How to implement a REST API?",
    "Explain the concept of Big O notation",
    "What are the principles of clean code?",
    "How do I use version control effectively?",
    "What's the difference between compiled and interpreted languages?",
    "How to implement unit testing?",
    "What are microservices and how do they work?",
    "How to handle concurrency in applications?",
    
    # Python (Specific)
    "How do I use list comprehensions in Python?",
    "What's the difference between lists and tuples in Python?",
    "How to handle file I/O in Python?",
    "Explain Python decorators with examples",
    "How do I use pandas for data analysis?",
    "What are Python generators and when to use them?",
    "How to create virtual environments in Python?",
    "Explain the GIL in Python",
    "How to use asyncio for asynchronous programming?",
    "What's the difference between __str__ and __repr__ in Python?",
    "How to implement a class in Python?",
    "How do I use matplotlib to create visualizations?",
    "What are Python lambda functions?",
    "How to handle JSON data in Python?",
    "How to use Flask to build a web application?",
    
    # Literature
    "Analyze the themes in Shakespeare's Hamlet",
    "What are the characteristics of Romantic poetry?",
    "Explain the symbolism in The Great Gatsby",
    "Compare and contrast Victorian and Modernist literature",
    "What is the significance of the green light in Gatsby?",
    "Analyze the character development in To Kill a Mockingbird",
    "What are the main themes in George Orwell's 1984?",
    "Explain the literary technique of stream of consciousness",
    "What is the role of the narrator in Jane Eyre?",
    "Discuss the use of allegory in Animal Farm",
    "What are the characteristics of Gothic literature?",
    "Analyze the feminist themes in The Handmaid's Tale",
    "What is the significance of the title Pride and Prejudice?",
    "Explain the concept of magical realism in literature",
    "What are the main themes in Beloved by Toni Morrison?",
    
    # Conversational
    "How are you doing today?",
    "What's your favorite movie?",
    "Tell me about your weekend plans",
    "What do you think about the weather?",
    "How's your family doing?",
    "What are your hobbies?",
    "Tell me a joke",
    "What's on your mind?",
    "How was your day?",
    "What's your opinion on social media?",
    "Do you have any pets?",
    "What's your favorite food?",
    "Tell me about yourself",
    "What are you up to?",
    "How's work going?",
    "What did you do last night?",
    "Any plans for the holidays?",
    "What's new with you?",
    "How's life treating you?",
    "What's your favorite book?",
    
    # Machine Learning
    "How do neural networks work?",
    "What's the difference between supervised and unsupervised learning?",
    "Explain gradient descent algorithm",
    "What are the types of machine learning algorithms?",
    "How does backpropagation work?",
    "What is overfitting in machine learning?",
    "Explain the concept of feature engineering",
    "What are decision trees and how do they work?",
    "How do you evaluate a machine learning model?",
    "What is the difference between classification and regression?",
    "Explain cross-validation in machine learning",
    "What are ensemble methods in ML?",
    "How does deep learning differ from traditional ML?",
    "What is transfer learning?",
    "Explain the concept of dimensionality reduction",
    "What are GANs and how do they work?",
    "How do you handle imbalanced datasets?",
    "What is reinforcement learning?",
    "Explain the bias-variance tradeoff",
    "What are convolutional neural networks?"
]'''

'''test = {
    "programming": [
        "def bubble_sort(arr): for i in range(len(arr)): for j in range(0, len(arr)-i-1):",
        "class DatabaseConnection: def __init__(self, host, port): self.connection = None",
        "try: result = api_call() except ConnectionError as e: logger.error(f'Failed: {e}')",
        "async function fetchData() { const response = await fetch('/api/users'); return response.json(); }",
        "if __name__ == '__main__': parser = argparse.ArgumentParser(); args = parser.parse_args()"
    ],
    
    "python": [
        "import pandas as pd; df = pd.read_csv('data.csv'); df.groupby('category').mean()",
        "with open('file.txt', 'r') as f: lines = [line.strip() for line in f.readlines()]",
        "from sklearn.linear_model import LinearRegression; model = LinearRegression().fit(X, y)",
        "lambda x: x**2 if x > 0 else -x**2; list(map(lambda x: x*2, [1,2,3,4,5]))",
        "import numpy as np; arr = np.random.randn(100, 50); eigenvals, eigenvecs = np.linalg.eig(arr)"
    ],
    
    "literature": [
        "The autumn leaves danced in the crisp morning air, whispering secrets of seasons past.",
        "She gazed upon the manuscript, its yellowed pages holding centuries of forgotten wisdom.",
        "His voice trembled with emotion as he recounted the tale of love lost and dreams shattered.",
        "The protagonist's journey through the labyrinthine city mirrored her own internal struggles.",
        "In the shadowy alcoves of the ancient library, time seemed to stand perfectly still."
    ],
    
    "conversational": [
        "Hey! How's your day going? I hope you're having a great time with your family.",
        "That sounds really interesting! Could you tell me more about what you're working on?",
        "I totally understand what you mean. I've been in a similar situation before myself.",
        "Thanks so much for your help! I really appreciate you taking the time to explain this.",
        "What do you think about trying that new restaurant downtown? I heard great reviews!"
    ],
    
    "machine_learning": [
        "The convolutional neural network achieved 94% accuracy on the ImageNet validation dataset.",
        "Gradient descent optimization with momentum converged faster than standard SGD approaches.",
        "Feature engineering and dimensionality reduction improved model performance significantly on high-dimensional data.",
        "Cross-validation revealed overfitting in the random forest model with default hyperparameters.",
        "The transformer architecture's self-attention mechanism enables parallel processing of sequential data."
    ]
}


data=[
# Programming (10 examples)
"for (int i = 0; i < array.length; i++) { sum += array[i]; }",
"function calculateDistance(x1, y1, x2, y2) { return Math.sqrt((x2-x1)**2 + (y2-y1)2); }",
"while (queue.isEmpty() == false) { node = queue.dequeue(); processNode(node); }",
"if (user.isAuthenticated()) { redirectTo('/dashboard'); } else { showLoginForm(); }",
"const hashMap = new Map(); hashMap.set(key, value); return hashMap.get(key);",
"recursiveFunction(n) { if (n <= 1) return 1; else return n * recursiveFunction(n-1); }",
"try { connection.execute(query); } catch (SQLException e) { rollback(); }",
"struct Node { int data; Node next; Node prev; };",
"SELECT users.name, orders.total FROM users JOIN orders ON users.id = orders.user_id",
"algorithm quickSort(arr, low, high): if low < high then partition and recurse",
# Python (10 examples)  
"import matplotlib.pyplot as plt; plt.plot(x_data, y_data); plt.show()",
"df['new_column'] = df.apply(lambda row: row['col1'] + row['col2'], axis=1)",
"with sqlite3.connect('database.db') as conn: cursor = conn.execute(query)",
"class Person: def __init__(self, name, age): self.name = name; self.age = age",
"[x**2 for x in range(10) if x % 2 == 0]",
"import requests; response = requests.get('https://api.example.com/data')",
"from collections import defaultdict; word_count = defaultdict(int)",
"np.where(array > threshold, array, 0)",
"pickle.dump(model, open('trained_model.pkl', 'wb'))",
"async def fetch_data(): async with aiohttp.ClientSession() as session:",

# Literature (10 examples)
"The moonlight cascaded through the ancient oak's gnarled branches, casting ethereal shadows.",
"Her heart ached with the weight of unspoken words and memories long buried.",
"In the distance, the cathedral bells echoed across the cobblestone streets of the old city.",
"Time seemed to slow as she turned the yellowed pages of her grandmother's diary.",
"The storm raged outside while inside, by the fireplace, stories came alive through whispered words.",
"His eyes held the depth of oceans and the mystery of forgotten civilizations.",
"The garden bloomed with roses that carried secrets of love letters never sent.",
"Through the mist emerged a figure cloaked in velvet, walking toward an uncertain destiny.",
"The library's silence was broken only by the gentle rustling of ancient manuscripts.",
"She painted her sorrows in watercolors that bled like tears across the canvas.",

# Conversational (10 examples)
"Hey there! How was your weekend? Did you get up to anything fun?",
"That's awesome! I'd love to hear more about your trip to Japan sometime.",
"No worries at all! Thanks for letting me know you're running a bit late.",
"What do you think about grabbing coffee this afternoon if you're free?",
"I totally get what you mean - I've been in the exact same situation before.",
"Congrats on the promotion! You really deserve it after all your hard work.",
"Hope you're feeling better today! Let me know if you need anything at all.",
"That sounds like such a great idea! Count me in if you need an extra person.",
"Thanks so much for helping me out with this - I really appreciate your time.",
"How's your family doing? I hope everyone is staying healthy and happy.",

# Machine Learning (10 examples)
"The neural network achieved 97.2% accuracy on the test dataset after hyperparameter tuning.",
"Support vector machines with RBF kernels performed better than linear classifiers on this problem.",
"Cross-validation revealed significant overfitting when using too many hidden layers.",
"Feature selection using mutual information reduced dimensionality from 10,000 to 500 features.",
"The random forest model showed lower variance compared to individual decision trees.",
"Batch normalization improved convergence speed during backpropagation training.",
"Transfer learning from pretrained ImageNet models accelerated training on our custom dataset.",
"The attention mechanism in transformers enables parallel processing of sequence data.",
"Regularization techniques like dropout and L2 penalty reduced model overfitting significantly.",
"Ensemble methods combining multiple weak learners outperformed single strong classifiers."]'''

'''# Reference strings for domain classification
test = {
    "programming": [
        "What are the best practices for code refactoring?",
        "How do I implement a hash table from scratch?",
        "Explain the differences between functional and imperative programming"
    ],
    
    "python": [
        "How do I work with virtual environments in Python?",
        "What's the purpose of decorators in Python?",
        "How to use enumerate() and zip() functions effectively?"
    ],
    
    "literature": [
        "What are the main motifs in To Kill a Mockingbird?",
        "How does magical realism function in One Hundred Years of Solitude?",
        "Discuss the narrative structure of Wuthering Heights"
    ],
    
    "conversational": [
        "What did you think of that new series on Netflix?",
        "Any exciting plans for this weekend?",
        "Hope you had a good lunch break!"
    ],
    
    "machine_learning": [
        "What is the role of activation functions in neural networks?",
        "How does random forest prevent overfitting?",
        "What are the advantages of ensemble learning methods?"
    ]
}

# Test dataset for domain classification
data = [
    # Programming (10 examples)
    "public class LinkedList { private Node head; public void insert(int data) { Node newNode = new Node(data); } }",
    "def merge_sort(arr): if len(arr) <= 1: return arr; mid = len(arr) // 2; left = merge_sort(arr[:mid])",
    "switch (operation) { case 'ADD': result = a + b; break; case 'SUBTRACT': result = a - b; break; }",
    "CREATE TABLE customers (id INT PRIMARY KEY, name VARCHAR(100), email VARCHAR(255) UNIQUE);",
    "function debounce(func, delay) { let timeout; return function() { clearTimeout(timeout); timeout = setTimeout(func, delay); } }",
    "interface Shape { double calculateArea(); double calculatePerimeter(); }",
    "git checkout -b feature/new-authentication; git add .; git commit -m 'Add OAuth integration'",
    "const express = require('express'); const app = express(); app.get('/api/users', (req, res) => { res.json(users); });",
    "binary_search(array, target, left=0, right=len(array)-1): if left > right: return -1",
    "void* malloc(size_t size); free(ptr); // Dynamic memory allocation in C",
    
    # Python (10 examples)  
    "from datetime import datetime; now = datetime.now(); formatted = now.strftime('%Y-%m-%d %H:%M:%S')",
    "import json; with open('config.json', 'r') as f: config = json.load(f)",
    "from flask import Flask, render_template; app = Flask(__name__); @app.route('/') def home(): return render_template('index.html')",
    "try: age = int(input('Enter age: ')); except ValueError: print('Invalid input')",
    "import os; files = [f for f in os.listdir('.') if f.endswith('.py')]",
    "from threading import Thread; def worker(): pass; t = Thread(target=worker); t.start()",
    "import re; pattern = r'\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b'; matches = re.findall(pattern, text)",
    "def fibonacci(n): a, b = 0, 1; for _ in range(n): yield a; a, b = b, a + b",
    "from itertools import combinations; pairs = list(combinations(items, 2))",
    "import configparser; config = configparser.ConfigParser(); config.read('settings.ini')",
    
    # Literature (10 examples)
    "The fog rolled in silently, wrapping the harbor in a blanket of mystery and solitude.",
    "She traced her fingers along the spines of leather-bound volumes, each one a doorway to another world.",
    "The old grandfather clock chimed midnight as rain began to tap against the windowpanes.",
    "His words hung in the air like delicate snowflakes, beautiful but destined to melt away.",
    "The cottage stood alone on the hill, its windows glowing warmly against the approaching dusk.",
    "Memory is a cruel curator, preserving only fragments while letting treasures slip into darkness.",
    "The lighthouse beam swept across the rocky coastline, a steadfast guardian against the night.",
    "Her laughter echoed through the empty hallways, a ghost of happier times long past.",
    "The merchant's tales were woven with threads of truth and fantasy, impossible to separate.",
    "Dawn broke slowly over the moors, painting the heather in shades of gold and crimson.",
    
    # Conversational (10 examples)
    "Good morning! Did you sleep well? I heard there was quite a storm last night.",
    "That restaurant recommendation was perfect! The pasta was absolutely incredible.",
    "I'm so sorry I missed your call earlier - my phone was on silent during the meeting.",
    "Would you be interested in joining our book club? We meet every Thursday evening.",
    "I can totally relate to that feeling - starting a new job can be both exciting and nerve-wracking.",
    "Happy birthday! I hope you have the most wonderful day celebrating with loved ones.",
    "Take your time with that decision - there's no rush and I'm here if you need to talk through it.",
    "That vacation photo looks amazing! The sunset behind those mountains is absolutely breathtaking.",
    "You've been such a great friend through all of this - I don't know what I would have done without you.",
    "Have you heard anything about the weather forecast for this weekend? I'm hoping it stays sunny.",
    
    # Machine Learning (10 examples)
    "K-means clustering partitioned the dataset into five distinct groups based on feature similarity.",
    "The LSTM network successfully captured long-term dependencies in the sequential time series data.",
    "Principal component analysis reduced the feature space while preserving 95% of the original variance.",
    "Precision and recall metrics indicated that the model performed well on minority class detection.",
    "Data augmentation techniques increased the training dataset size by 300% without overfitting issues.",
    "The learning rate scheduler gradually decreased the step size to improve convergence stability.",
    "Confusion matrix analysis revealed that the classifier struggled most with distinguishing between classes 2 and 4.",
    "Early stopping prevented overfitting by monitoring validation loss throughout the training process.",
    "The recommender system achieved a 15% improvement in user engagement using collaborative filtering.",
    "Hyperparameter optimization with Bayesian search found the optimal configuration in fewer iterations."
]'''

'''# Reference strings for domain classification
test = {
    "programming": [
        "What are the best practices for code refactoring?",
        "How do I implement a hash table from scratch?",
        "Explain the differences between functional and imperative programming"
    ],
    
    "python": [
        "How do I work with virtual environments in Python?",
        "What's the purpose of decorators in Python?",
        "How to use enumerate() and zip() functions effectively?"
    ],
    
    "literature": [
        "What are the main motifs in To Kill a Mockingbird?",
        "How does magical realism function in One Hundred Years of Solitude?",
        "Discuss the narrative structure of Wuthering Heights"
    ],
    
    "conversational": [
        "What did you think of that new series on Netflix?",
        "Any exciting plans for this weekend?",
        "Hope you had a good lunch break!"
    ],
    
    "machine_learning": [
        "What is the role of activation functions in neural networks?",
        "How does random forest prevent overfitting?",
        "What are the advantages of ensemble learning methods?"
    ]
}

# Extended test dataset for domain classification (100 examples, 20 per category)
data = [
    # Programming (20 examples)
    "public class LinkedList { private Node head; public void insert(int data) { Node newNode = new Node(data); } }",
    "def merge_sort(arr): if len(arr) <= 1: return arr; mid = len(arr) // 2; left = merge_sort(arr[:mid])",
    "switch (operation) { case 'ADD': result = a + b; break; case 'SUBTRACT': result = a - b; break; }",
    "CREATE TABLE customers (id INT PRIMARY KEY, name VARCHAR(100), email VARCHAR(255) UNIQUE);",
    "function debounce(func, delay) { let timeout; return function() { clearTimeout(timeout); timeout = setTimeout(func, delay); } }",
    "interface Shape { double calculateArea(); double calculatePerimeter(); }",
    "git checkout -b feature/new-authentication; git add .; git commit -m 'Add OAuth integration'",
    "const express = require('express'); const app = express(); app.get('/api/users', (req, res) => { res.json(users); });",
    "binary_search(array, target, left=0, right=len(array)-1): if left > right: return -1",
    "void* malloc(size_t size); free(ptr); // Dynamic memory allocation in C",
    "while (current != null) { if (current.data == target) return current; current = current.next; }",
    "int factorial(int n) { return (n <= 1) ? 1 : n * factorial(n - 1); }",
    "SELECT u.name, COUNT(o.id) as order_count FROM users u LEFT JOIN orders o ON u.id = o.user_id GROUP BY u.id",
    "docker run -p 8080:80 nginx:latest",
    "try { result = performOperation(); } catch (Exception e) { logger.error('Operation failed', e); }",
    "struct TreeNode { int val; TreeNode* left; TreeNode* right; TreeNode(int x) : val(x), left(NULL), right(NULL) {} };",
    "const API_URL = 'https://api.example.com'; fetch(API_URL + '/data').then(response => response.json())",
    "for i in range(len(matrix)): for j in range(len(matrix[i])): if matrix[i][j] == target: return (i, j)",
    "public static void quickSort(int[] arr, int low, int high) { if (low < high) { int pi = partition(arr, low, high); } }",
    "UPDATE products SET price = price * 1.1 WHERE category = 'electronics' AND stock > 0",
    
    # Python (20 examples)
    "from datetime import datetime; now = datetime.now(); formatted = now.strftime('%Y-%m-%d %H:%M:%S')",
    "import json; with open('config.json', 'r') as f: config = json.load(f)",
    "from flask import Flask, render_template; app = Flask(__name__); @app.route('/') def home(): return render_template('index.html')",
    "try: age = int(input('Enter age: ')); except ValueError: print('Invalid input')",
    "import os; files = [f for f in os.listdir('.') if f.endswith('.py')]",
    "from threading import Thread; def worker(): pass; t = Thread(target=worker); t.start()",
    "import re; pattern = r'\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b'; matches = re.findall(pattern, text)",
    "def fibonacci(n): a, b = 0, 1; for _ in range(n): yield a; a, b = b, a + b",
    "from itertools import combinations; pairs = list(combinations(items, 2))",
    "import configparser; config = configparser.ConfigParser(); config.read('settings.ini')",
    "import sqlite3; conn = sqlite3.connect('database.db'); cursor = conn.cursor(); cursor.execute('SELECT * FROM users')",
    "from collections import defaultdict, Counter; word_counts = Counter(words); freq_dict = defaultdict(int)",
    "import pandas as pd; df = pd.read_csv('data.csv'); grouped = df.groupby('category').agg({'price': 'mean'})",
    "from pathlib import Path; home = Path.home(); files = list(home.glob('*.txt'))",
    "import asyncio; async def fetch_data(): await asyncio.sleep(1); return 'data'",
    "from dataclasses import dataclass; @dataclass class Person: name: str; age: int",
    "import pickle; with open('model.pkl', 'wb') as f: pickle.dump(trained_model, f)",
    "from typing import List, Dict, Optional; def process_data(items: List[str]) -> Dict[str, int]: return {}",
    "import requests; response = requests.post('https://api.example.com/submit', json={'key': 'value'})",
    "from functools import lru_cache; @lru_cache(maxsize=128) def expensive_function(n): return n ** 2",
    
    # Literature (20 examples)
    "The fog rolled in silently, wrapping the harbor in a blanket of mystery and solitude.",
    "She traced her fingers along the spines of leather-bound volumes, each one a doorway to another world.",
    "The old grandfather clock chimed midnight as rain began to tap against the windowpanes.",
    "His words hung in the air like delicate snowflakes, beautiful but destined to melt away.",
    "The cottage stood alone on the hill, its windows glowing warmly against the approaching dusk.",
    "Memory is a cruel curator, preserving only fragments while letting treasures slip into darkness.",
    "The lighthouse beam swept across the rocky coastline, a steadfast guardian against the night.",
    "Her laughter echoed through the empty hallways, a ghost of happier times long past.",
    "The merchant's tales were woven with threads of truth and fantasy, impossible to separate.",
    "Dawn broke slowly over the moors, painting the heather in shades of gold and crimson.",
    "The ancient oak stretched its gnarled branches toward heaven, bearing witness to centuries of secrets.",
    "Shadows danced upon the castle walls as candlelight flickered in the great hall.",
    "The river whispered its eternal song as it wound through valleys forgotten by time.",
    "She opened the dusty tome, and the scent of ages past rose from its yellowed pages.",
    "The wind carried stories across the meadow, each blade of grass a keeper of memories.",
    "In the depths of winter, hope bloomed like a rare flower in her frost-touched heart.",
    "The cathedral's spires pierced the clouds, reaching toward divine mysteries beyond mortal understanding.",
    "His pen moved across parchment like a sword through silk, carving truth from imagination.",
    "The labyrinth of narrow streets held secrets in every shadowed doorway and weathered stone.",
    "Time flowed differently in the enchanted grove, where fairy rings marked ancient gatherings.",
    
    # Conversational (20 examples)
    "Good morning! Did you sleep well? I heard there was quite a storm last night.",
    "That restaurant recommendation was perfect! The pasta was absolutely incredible.",
    "I'm so sorry I missed your call earlier - my phone was on silent during the meeting.",
    "Would you be interested in joining our book club? We meet every Thursday evening.",
    "I can totally relate to that feeling - starting a new job can be both exciting and nerve-wracking.",
    "Happy birthday! I hope you have the most wonderful day celebrating with loved ones.",
    "Take your time with that decision - there's no rush and I'm here if you need to talk through it.",
    "That vacation photo looks amazing! The sunset behind those mountains is absolutely breathtaking.",
    "You've been such a great friend through all of this - I don't know what I would have done without you.",
    "Have you heard anything about the weather forecast for this weekend? I'm hoping it stays sunny.",
    "Oh wow, congratulations on your promotion! You totally deserve it after all your hard work.",
    "Thanks for picking up coffee for me this morning - you're such a lifesaver!",
    "How's your mom doing after her surgery? I've been thinking about her and hoping she's recovering well.",
    "Did you catch the game last night? What an incredible finish - I couldn't believe that last-minute goal!",
    "I'm really looking forward to our dinner plans tomorrow. Should we meet at the restaurant or do you want me to pick you up?",
    "That's such exciting news about your engagement! I'm so happy for you both - when's the big day?",
    "No worries about being late - traffic can be absolutely crazy this time of day.",
    "Your garden looks absolutely beautiful this year! Those roses are particularly stunning.",
    "I hope your presentation went well today. You were so prepared, I'm sure you knocked it out of the park!",
    "Thanks for listening to me vent about work stuff. Sometimes you just need a good friend to talk things through with.",
    
    # Machine Learning (20 examples)
    "K-means clustering partitioned the dataset into five distinct groups based on feature similarity.",
    "The LSTM network successfully captured long-term dependencies in the sequential time series data.",
    "Principal component analysis reduced the feature space while preserving 95% of the original variance.",
    "Precision and recall metrics indicated that the model performed well on minority class detection.",
    "Data augmentation techniques increased the training dataset size by 300% without overfitting issues.",
    "The learning rate scheduler gradually decreased the step size to improve convergence stability.",
    "Confusion matrix analysis revealed that the classifier struggled most with distinguishing between classes 2 and 4.",
    "Early stopping prevented overfitting by monitoring validation loss throughout the training process.",
    "The recommender system achieved a 15% improvement in user engagement using collaborative filtering.",
    "Hyperparameter optimization with Bayesian search found the optimal configuration in fewer iterations.",
    "Convolutional neural networks with attention mechanisms outperformed traditional CNN architectures on image classification.",
    "The transformer model's self-attention weights revealed which input tokens were most relevant for each prediction.",
    "Regularization techniques including dropout and batch normalization significantly reduced model variance.",
    "Cross-validation with stratified sampling ensured balanced representation across all target classes.",
    "The ensemble method combined predictions from random forest, gradient boosting, and neural network models.",
    "Feature importance analysis using SHAP values identified the top predictive variables in the dataset.",
    "The autoencoder successfully learned compressed representations that preserved essential data characteristics.",
    "Transfer learning from pre-trained models accelerated convergence on the domain-specific classification task.",
    "Adversarial training improved model robustness against input perturbations and edge cases.",
    "The reinforcement learning agent optimized its policy through trial-and-error interaction with the environment."
]'''

# Content-based reference strings for domain classification
test = {
    "programming": [
        "public class BinaryTree { private Node root; public void insert(int value) { root = insertRec(root, value); } }",
        "function quickSort(arr, low, high) { if (low < high) { let pi = partition(arr, low, high); quickSort(arr, low, pi - 1); } }",
        "struct LinkedList { int data; struct LinkedList* next; }; void insertNode(struct LinkedList** head, int value) { }"
    ],
    
    "python": [
        "import pandas as pd; df = pd.read_csv('data.csv'); result = df.groupby('category').agg({'price': 'mean', 'quantity': 'sum'})",
        "from sklearn.model_selection import train_test_split; X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)",
        "with open('config.json', 'r') as f: config = json.load(f); app = Flask(__name__); app.config.update(config)"
    ],
    
    "literature": [
        "The amber light of late afternoon filtered through the dusty windows, casting long shadows across the abandoned library.",
        "Her heart carried the weight of unspoken words, each memory a delicate thread in the tapestry of her past.",
        "The cathedral bells rang out across the cobblestone square, their resonant tones echoing through centuries of history."
    ],
    
    "conversational": [
        "Hey! How's your day going? I hope you're having a wonderful time with your family this weekend.",
        "That sounds absolutely fascinating! I'd love to hear more about your recent trip to Italy when you have time.",
        "Thanks so much for your help earlier - I really appreciate you taking the time to walk me through everything."
    ],
    
    "machine_learning": [
        "The convolutional neural network achieved 94.7% accuracy on the validation set after implementing data augmentation and dropout regularization.",
        "Cross-validation results showed that the random forest model with 100 estimators outperformed both SVM and logistic regression on this dataset.",
        "The transformer architecture's self-attention mechanism allows for parallel processing of sequence data, significantly reducing training time."
    ]
}

# Question-based test dataset for domain classification
data = [
    # Programming (20 examples)
    "How do I implement a binary search algorithm efficiently?",
    "What's the difference between stack and heap memory allocation?",
    "Can you explain object-oriented programming principles?",
    "How should I optimize database queries for better performance?",
    "What are design patterns and when should I use them?",
    "How do I properly handle exceptions in my code?",
    "What's the best way to structure a large codebase?",
    "How can I implement a RESTful API effectively?",
    "Can you explain the concept of Big O notation?",
    "What are the key principles of writing clean code?",
    "How do I use version control systems effectively?",
    "What's the difference between compiled and interpreted languages?",
    "How should I approach implementing unit tests?",
    "What are microservices and how do they work?",
    "How can I handle concurrency in my applications?",
    "What are the best practices for code refactoring?",
    "How do I implement a hash table from scratch?",
    "What's the difference between functional and imperative programming?",
    "How can I improve the performance of my algorithms?",
    "What are the principles of good software architecture?",
    
    # Python (20 examples)
    "How do I use list comprehensions in Python effectively?",
    "What's the difference between lists and tuples in Python?",
    "How should I handle file I/O operations in Python?",
    "Can you explain Python decorators with examples?",
    "How do I use pandas for data analysis tasks?",
    "What are Python generators and when should I use them?",
    "How can I create and manage virtual environments in Python?",
    "Can you explain the Global Interpreter Lock (GIL) in Python?",
    "How do I use asyncio for asynchronous programming?",
    "What's the difference between __str__ and __repr__ in Python?",
    "How should I implement classes in Python?",
    "How can I use matplotlib to create effective visualizations?",
    "What are Python lambda functions and how do I use them?",
    "How should I handle JSON data in Python applications?",
    "How can I use Flask to build a web application?",
    "What are the best practices for Python package management?",
    "How do I work with virtual environments in Python?",
    "What's the purpose of decorators in Python?",
    "How can I use enumerate() and zip() functions effectively?",
    "What are the best practices for Python testing?",
    
    # Literature (20 examples)
    "What are the main themes in Shakespeare's Hamlet?",
    "How does symbolism function in The Great Gatsby?",
    "Can you analyze the character development in To Kill a Mockingbird?",
    "What are the characteristics of Romantic poetry?",
    "How does magical realism work in One Hundred Years of Solitude?",
    "What is the significance of the narrative structure in Wuthering Heights?",
    "Can you explain the use of allegory in Animal Farm?",
    "What are the main themes in George Orwell's 1984?",
    "How does the stream of consciousness technique work in literature?",
    "What is the role of the narrator in Jane Eyre?",
    "Can you analyze the feminist themes in The Handmaid's Tale?",
    "What is the significance of the title Pride and Prejudice?",
    "How does irony function in Jane Austen's novels?",
    "What are the main themes in Beloved by Toni Morrison?",
    "How do you analyze poetic meter and rhythm?",
    "What are the characteristics of Gothic literature?",
    "How does foreshadowing work in mystery novels?",
    "What are the main motifs in To Kill a Mockingbird?",
    "How does point of view affect storytelling?",
    "What is the significance of setting in literature?",
    
    # Conversational (20 examples)
    "How are you doing today?",
    "What's your favorite movie of all time?",
    "Can you tell me about your weekend plans?",
    "What do you think about the current weather?",
    "How's your family doing these days?",
    "What are some of your favorite hobbies?",
    "Can you tell me a good joke?",
    "What's currently on your mind?",
    "How was your day at work today?",
    "What's your opinion on social media platforms?",
    "Do you have any pets at home?",
    "What's your favorite type of food?",
    "Can you tell me a bit about yourself?",
    "What are you up to this evening?",
    "How's work been treating you lately?",
    "What did you do last night for fun?",
    "Do you have any plans for the upcoming holidays?",
    "What's new and exciting in your life?",
    "How has life been treating you recently?",
    "What's your favorite book you've read lately?",
    
    # Machine Learning (20 examples)
    "How do neural networks actually work?",
    "What's the difference between supervised and unsupervised learning?",
    "Can you explain the gradient descent algorithm?",
    "What are the main types of machine learning algorithms?",
    "How does backpropagation work in neural networks?",
    "What is overfitting and how can I prevent it?",
    "Can you explain the concept of feature engineering?",
    "How do decision trees work for classification?",
    "What are the best ways to evaluate machine learning models?",
    "What's the difference between classification and regression?",
    "How does cross-validation work in machine learning?",
    "What are ensemble methods and when should I use them?",
    "How does deep learning differ from traditional machine learning?",
    "What is transfer learning and how do I apply it?",
    "Can you explain dimensionality reduction techniques?",
    "What are GANs and how do they work?",
    "How should I handle imbalanced datasets?",
    "What is reinforcement learning and its applications?",
    "Can you explain the bias-variance tradeoff?",
    "How do convolutional neural networks work for image recognition?"
]

dataAnalysis=DataAnalysis(test)
start=time.time()
outputs=dataAnalysis.getEncoding(data)
end=time.time()
errorCount=0
for i,output in enumerate(outputs):
    outputList=output.tolist()
    domainIndex=outputList.index(max(outputList))
    print(f"Predicted: {list(test.keys())[domainIndex]}, Actual: {list(test.keys())[i//20]}")
    if list(test.keys())[domainIndex]!=list(test.keys())[i//20]:
        errorCount+=1
print(f"Total errors: {errorCount} out of {len(data)}")
print(f"Error rate: {errorCount/len(data)*100:.2f}%")
print(f"Encoding time: {end-start} seconds")