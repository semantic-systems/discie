import matplotlib.pyplot as plt

F1 = 74.97
mention_F1 = 85.51
candidate_F1 = 86.91
ranking_F1 = 90.03
property_F1 = 100



difference_mention = mention_F1 - F1
difference_candidate = candidate_F1 - mention_F1
difference_ranking = ranking_F1 - candidate_F1
difference_property = property_F1 - ranking_F1


# Create pie chart
labels = ['Mention\nrecognition', 'Candidate\ngeneration', 'Candidate\nranking', 'Relation\nextraction']
sizes = [difference_mention/(100 - F1), difference_candidate/(100 - F1), difference_ranking/(100 - F1), difference_property/(100 - F1)]
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
# explode = (0.05,0.05,0.05,0.05)
fig, ax = plt.subplots()
# ax.pie(sizes, labels=labels)
# Give percentages for each label in each slice
wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, pctdistance=0.6)

plt.savefig('pie.png')



