# def question_3(mountains):
#     L = []
#     R = []
#     for i in range(len(mountains)):
#         left_mountain = (mountains[0: i])
#         left_mountain.reverse()
#         right_mountain = mountains[i + 1:]
#         if len(left_mountain) == 0:
#             L.append("None")
#         else:
#             for index, value in enumerate(left_mountain):
#                 if value > mountains[i]:
#                     L.append(index)
#                     break
#         if len(right_mountain) == 0:
#             R.append("None")
#         else:
#             for index, value in enumerate(right_mountain):
#                 if value > mountains[i]:
#                     R.append(index + 1)
#                     break
#     return L, R

#
# def question_3(mountains):
#     L = []
#     R = []
#     for i in range(len(mountains)):
#         p_l, p_r = i, i
#         while True:
#             p_l -= 1
#             if p_l == -1:
#                 L.append(None)
#                 break
#             elif mountains[p_l] > mountains[i]:
#                 L.append(p_l)
#                 break
#         while True:
#             p_r += 1
#             if p_r == len(mountains):
#                 R.append(None)
#                 break
#             elif mountains[p_r] > mountains[i]:
#                 R.append(p_r)
#                 break
#     return L, R
#
#
# A1 = [5, 2, 6, 8, 1, 4, 3, 9]
# L, R = question_3(A1)
# print(L, R)


from sklearn.metrics import average_precision_score, precision_score
from sklearn.metrics import recall_score

true_y = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1]
predict_y = [1, 2, 3, 1, 3, 2, 3, 2, 1, 3]

precision = precision_score(true_y, predict_y, average="macro", labels=[3])
recall = recall_score(true_y, predict_y, average="macro", labels=[2])
print(precision)
print(recall)
