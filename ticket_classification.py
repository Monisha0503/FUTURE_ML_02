# Task 2 - Support Ticket Classification
# Future Interns - FUTURE_ML_02

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load data
df = pd.read_csv('customer_support_tickets.csv')
print("Shape:", df.shape)
print("\nTicket Types:")
print(df['Ticket Type'].value_counts())
print("\nTicket Priority:")
print(df['Ticket Priority'].value_counts())

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
df['Ticket Type'].value_counts().plot(kind='bar', ax=axes[0],
                                       color='skyblue', edgecolor='black')
axes[0].set_title('Ticket Type Distribution', fontsize=14)
axes[0].tick_params(axis='x', rotation=45)
df['Ticket Priority'].value_counts().plot(kind='bar', ax=axes[1],
                                           color='orange', edgecolor='black')
axes[1].set_title('Ticket Priority Distribution', fontsize=14)
plt.tight_layout()
plt.savefig('ticket_distribution.png')
plt.show()

# ML Model
X = df['Ticket Subject'].fillna('')
y = df['Ticket Type']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
tfidf = TfidfVectorizer(max_features=500, stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)
print(f"\nModel Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print(classification_report(y_test, y_pred))

# Priority Tagging
def assign_priority(ticket_type):
    if ticket_type in ['Technical issue', 'Refund request']:
        return 'High'
    elif ticket_type in ['Billing inquiry', 'Cancellation request']:
        return 'Medium'
    else:
        return 'Low'

df['Predicted_Priority'] = df['Ticket Type'].apply(assign_priority)
print("\nPriority Distribution:")
print(df['Predicted_Priority'].value_counts())

# Final Graph
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
df['Predicted_Priority'].value_counts().plot(kind='pie',
    ax=axes[0], autopct='%1.1f%%',
    colors=['red', 'orange', 'green'])
axes[0].set_title('Predicted Priority Distribution', fontsize=14)
df.groupby(['Ticket Type', 'Predicted_Priority']).size().unstack().plot(
    kind='bar', ax=axes[1], colormap='Set2')
axes[1].set_title('Ticket Type vs Priority', fontsize=14)
axes[1].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('ticket_priority.png')
plt.show()
print("✅ Task 2 Complete!")
