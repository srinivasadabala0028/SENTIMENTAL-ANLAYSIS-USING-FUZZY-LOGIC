import nltk
nltk.download('vader_lexicon')
import pandas as pd
import re
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import time

start = time.time()

# Load data
traindata = pd.read_csv("/content/dataset (1).csv", encoding='ISO-8859-1')
doc = traindata.TweetText
sentidoc = traindata.Sentiment

# Generate universe variables
x_p = np.arange(0, 1, 0.1)
x_n = np.arange(0, 1, 0.1)
x_op = np.arange(0, 10, 1)

# Generate fuzzy membership functions
p_lo = fuzz.trimf(x_p, [0, 0, 0.5])
p_md = fuzz.trimf(x_p, [0, 0.5, 1])
p_hi = fuzz.trimf(x_p, [0.5, 1, 1])
n_lo = fuzz.trimf(x_n, [0, 0, 0.5])
n_md = fuzz.trimf(x_n, [0, 0.5, 1])
n_hi = fuzz.trimf(x_n, [0.5, 1, 1])
op_Neg = fuzz.trimf(x_op, [0, 0, 5])  # Scale: Neg Neu Pos
op_Neu = fuzz.trimf(x_op, [0, 5, 10])
op_Pos = fuzz.trimf(x_op, [5, 10, 10])

# Initialize sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Initialize lists to store sentiment results
sentiment = []

# Loop through each document
for j in range(len(doc)):
    # Preprocess tweet text
    tweet_text = re.sub(r"@", "", doc[j].lower())  # Remove @ and convert to lowercase

    # Perform sentiment analysis using VADER
    ss = sid.polarity_scores(tweet_text)
    posscore = ss['pos']
    negscore = ss['neg']

    # Calculate membership values for positive and negative sentiment
    p_level_lo = fuzz.interp_membership(x_p, p_lo, posscore)
    p_level_md = fuzz.interp_membership(x_p, p_md, posscore)
    p_level_hi = fuzz.interp_membership(x_p, p_hi, posscore)

    n_level_lo = fuzz.interp_membership(x_n, n_lo, negscore)
    n_level_md = fuzz.interp_membership(x_n, n_md, negscore)
    n_level_hi = fuzz.interp_membership(x_n, n_hi, negscore)

    # Define fuzzy rules
    active_rule1 = np.fmin(p_level_lo, n_level_lo)
    active_rule2 = np.fmin(p_level_md, n_level_lo)
    active_rule3 = np.fmin(p_level_hi, n_level_lo)
    active_rule4 = np.fmin(p_level_lo, n_level_md)
    active_rule5 = np.fmin(p_level_md, n_level_md)
    active_rule6 = np.fmin(p_level_hi, n_level_md)
    active_rule7 = np.fmin(p_level_lo, n_level_hi)
    active_rule8 = np.fmin(p_level_md, n_level_hi)
    active_rule9 = np.fmin(p_level_hi, n_level_hi)

    n1 = np.fmax(active_rule4, active_rule7)
    n2 = np.fmax(n1, active_rule8)
    op_activation_lo = np.fmin(n2, op_Neg)

    neu1 = np.fmax(active_rule1, active_rule5)
    neu2 = np.fmax(neu1, active_rule9)
    op_activation_md = np.fmin(neu2, op_Neu)

    p1 = np.fmax(active_rule2, active_rule3)
    p2 = np.fmax(p1, active_rule6)
    op_activation_hi = np.fmin(p2, op_Pos)

    # Aggregate output membership functions
    aggregated = np.fmax(op_activation_lo, np.fmax(op_activation_md, op_activation_hi))

    # Defuzzify to obtain final sentiment score
    op = fuzz.defuzz(x_op, aggregated, 'centroid')
    output = round(op, 2)

    # Determine sentiment label based on output score
    if 0 < output < 3.33:
        sentiment.append("Negative")
    elif 3.34 < output < 6.66:
        sentiment.append("Neutral")
    elif 6.67 < output < 10:
        sentiment.append("Positive")

    # Visualization
    fig, ax0 = plt.subplots(figsize=(8, 3))

    ax0.fill_between(x_op, np.zeros_like(x_op), op_activation_lo, facecolor='b', alpha=0.7)
    ax0.plot(x_op, op_Neg, 'b', linewidth=0.5, linestyle='--', label='Negative')
    ax0.fill_between(x_op, np.zeros_like(x_op), op_activation_md, facecolor='g', alpha=0.7)
    ax0.plot(x_op, op_Neu, 'g', linewidth=0.5, linestyle='--', label='Neutral')
    ax0.fill_between(x_op, np.zeros_like(x_op), op_activation_hi, facecolor='r', alpha=0.7)
    ax0.plot(x_op, op_Pos, 'r', linewidth=0.5, linestyle='--', label='Positive')
    ax0.plot([op, op], [0, fuzz.interp_membership(x_op, aggregated, op)], 'k', linewidth=1.5, alpha=0.9)
    ax0.set_title('Output membership activity')
    ax0.legend()

end = time.time()
print("Execution Time: " + str(round((end - start), 3)) + " secs")
