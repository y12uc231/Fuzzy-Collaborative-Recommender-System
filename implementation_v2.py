# Developer: Satyapriya Krishna
# University ID: Y12UC231
# Paper: Fuzzy Genetic Recommender System
# Packages: Pandas,Numpy
# Total Running Time: 33+-5 minutes
# item_rating1.csv = Item dataset(MovieLens)
# final_rating.csv = Rating Dataset(MovieLens)
# For generic code, generes are named as "G1","G2","G3",etc. We have extended to 20 genres( 2 extras) to show that the code is easily extendable

#####********************************************************###############################

import pandas as pd
import numpy as np


# Functions
def euclidean(a, b):
    a = np.array(a[0])
    b = np.array(b[0])

    l1 = len(a)
    # TEST::print l1
    sum1 = 0
    for i in range(0, l1, 1):
        # TEST:::print a[i],b[i]
        sum1 += pow((a[i] - b[i]), 2)
    sum1 = pow(sum1, 1.0 / 2)
    return sum1


def predict(gim, i, k, nbd, rated_members, res, item):
    pri = gim[gim.user_id == i].mean_rating.values[0]
    diff = 0
    for count in range(len(rated_members)):
        diff += (nbd[nbd.n_id == rated_members[count]].distance.values[0]) * (
        res[(res.user_id == rated_members[count]) & (res.item_id == item)].rating.values[0] -
        gim[gim.user_id == rated_members[count]].mean_rating.values[0])
    diff *= k
    pri += diff
    return pri


def calculate_mae(mov):
    sum1 = 0
    mov1 = mov.values
    for i in range(len(mov1)):
        if np.isnan(mov1[i, 2]) == False:
            sum1 += abs(mov1[i, 1] - mov1[i, 2])
    avg = float(sum1) / float(len(mov1))
    return avg


# Step1: Data Collection: Preparing the Consolidated Data


user = pd.read_csv('./data/final_rating.csv', sep=',')
del user['Index']
df1 = user
item = pd.read_csv('./data/item_rating1.csv', sep=',')
del item['Index']
df2 = item
# Final Database
result = df1.merge(df2, on='item_id', how='inner')
# Removing users who rated below 100 movies
count1 = result.user_id.value_counts()
count1.index.name = 'user_id'
count1 = pd.DataFrame(count1)
count1.columns = ['total_rating']
mask = count1.total_rating > 100  # Change Value as per req.. For ex: 100: users who rated more than 100 movies
temp1 = count1[mask]
c1 = temp1.index
c2 = temp1.total_rating
data = {'user_id': c1, 'total_rating': c2}
temp2 = pd.DataFrame(data)
result2 = result.merge(temp2, on='user_id', how='inner')  # 5 Genres(G1,G2,G3,G4,G5)

print("******************************Data Version-1*******************************")
print(result2.head())

write = result2  # Writing to a file
write.index.name = 'Index'
write['Occupation'] = np.random.randint(6, size=len(write))
write['Age'] = np.random.randint(100, size=len(write))

write.to_csv('./data/consolidate_v2.csv', sep=',')  # Writing the Consolidated

# Important Paramenters
genres = ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10', 'G11', 'G12', 'G13', 'G14', 'G15', 'G16', 'G17',
          'G18', 'G19', 'G20']
t_rating = len(result2)
user_freq = result2.user_id.value_counts()
user_vals = list(user_freq.index)
user_vals.sort()

# Step2: Hybrid Feature Creation : Adding GIM values in the 'refined1' DataFrame

refined1 = pd.DataFrame(
    columns=['user_id', 'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10', 'G11', 'G12', 'G13', 'G14', 'G15',
             'G16', 'G17', 'G18', 'G19', 'G20'])  # Stores GIM
col = np.array(result2.columns)
# TESTING: print user_vals
# p.append(dict,ignore_index=True) Adds a new row to the dataframe
grouped = result2.groupby('user_id')

print("Calculating GIMS.....")

for i in user_vals:
    # TESTCASE: print i
    r = {'user_id': i}
    for j in genres:
        group1 = grouped.get_group(i)
        val = group1.values
        TR = val[:, col == 'rating'].sum()

        temp_gr = group1[(group1[j] > 0) & (group1.rating >= 3)]
        GR = temp_gr.rating.sum()
        RGR = float(GR) / TR
        TF = len(group1)
        nf = float(TR) / TF
        group2 = temp_gr.groupby('rating')
        sum_num = 0
        count_group2 = group2.count()
        group2_vals = list(count_group2.index)
        for k in group2_vals:
            rating_group = group2.get_group(k)
            sum_num += ((k - 2) * len(rating_group))
        MGRF = float(sum_num) / (3 * TF)
        GIM = float(2 * nf * RGR * MGRF) / (RGR + MGRF)
        r[j] = GIM
    refined1 = refined1.append(r, ignore_index=True)

refined1['Age'] = np.random.randint(100, size=len(refined1))
refined1['Occupation'] = np.random.randint(6, size=len(refined1))
refined1['Gender'] = np.random.randint(1, size=len(refined1))

print("GIM Calculated !")
print("**********************************Data Version 2: Including GIMS*****************************************")
print(refined1.head())

# TESTCASE: Only run after the code above works without any errors with consistent results
# Extra Calculation
# ref_val= refined1.values
# for k in range(1,6,1):
# ref_val[:,k]=ref_val[:,k]/ref_val[:,k].max()
# refined2=pd.DataFrame(ref_val,columns=refined1.columns)

# print refined2

# Add two more attributes

# Step 3: Neighbourhood Selection

# Fuzzifying Data

columns_fuzzy = ['user_id'
    , 'G1_VB', 'G1_B', 'G1_AV', 'G1_G', 'G1_VG', 'G1_E'
    , 'G2_VB', 'G2_B', 'G2_AV', 'G2_G', 'G2_VG', 'G2_E'
    , 'G3_VB', 'G3_B', 'G3_AV', 'G3_G', 'G3_VG', 'G3_E'
    , 'G4_VB', 'G4_B', 'G4_AV', 'G4_G', 'G4_VG', 'G4_E'
    , 'G5_VB', 'G5_B', 'G5_AV', 'G5_G', 'G5_VG', 'G5_E'
    , 'G6_VB', 'G6_B', 'G6_AV', 'G6_G', 'G6_VG', 'G6_E'
    , 'G7_VB', 'G7_B', 'G7_AV', 'G7_G', 'G7_VG', 'G7_E'
    , 'G8_VB', 'G8_B', 'G8_AV', 'G8_G', 'G8_VG', 'G8_E'
    , 'G9_VB', 'G9_B', 'G9_AV', 'G9_G', 'G9_VG', 'G9_E'
    , 'G10_VB', 'G10_B', 'G10_AV', 'G10_G', 'G10_VG', 'G10_E'
    , 'G11_VB', 'G11_B', 'G11_AV', 'G11_G', 'G11_VG', 'G11_E'
    , 'G12_VB', 'G12_B', 'G12_AV', 'G12_G', 'G12_VG', 'G12_E'
    , 'G13_VB', 'G13_B', 'G13_AV', 'G13_G', 'G13_VG', 'G13_E'
    , 'G14_VB', 'G14_B', 'G14_AV', 'G14_G', 'G14_VG', 'G14_E'
    , 'G15_VB', 'G15_B', 'G15_AV', 'G15_G', 'G15_VG', 'G15_E'
    , 'G16_VB', 'G16_B', 'G16_AV', 'G16_G', 'G16_VG', 'G16_E'
    , 'G17_VB', 'G17_B', 'G17_AV', 'G17_G', 'G17_VG', 'G17_E'
    , 'G18_VB', 'G18_B', 'G18_AV', 'G18_G', 'G18_VG', 'G18_E'
    , 'G19_VB', 'G19_B', 'G19_AV', 'G19_G', 'G19_VG', 'G19_E'
    , 'G20_VB', 'G20_B', 'G20_AV', 'G20_G', 'G20_VG', 'G20_E', 'A_young', 'A_middle', 'A_old', 'Occupation', 'Gender']

fuzzy_vals = np.empty([len(refined1), len(columns_fuzzy)])
refined1_val = refined1.values
no_genre = 20  # 20 genres

# Fuzzyfing Data: Each Genres fuzzified to 6 Fuzzy Sets(VB,B,AV,G,VG,E)
print("Fuzzyfing Data.......")
for i in range(0, len(refined1_val), 1):
    print("Processing user: ", i, "....")
    temp_fuzz = []
    temp_fuzz.extend([refined1_val[i, 0]])
    for j in range(1, no_genre + 1, 1):  # For 5 Features
        # VB
        c_temp = 2
        if refined1_val[i, j] <= 1:
            G_VB = 1 - refined1_val[i, j]
        elif refined1_val[i, j] > 1:
            G_VB = 0
        temp_fuzz.extend([G_VB])

        # B,AV,G,VG
        if refined1_val[i, j] <= (c_temp - 2) or refined1_val[i, j] > c_temp:
            G_B = 0
        elif refined1_val[i, j] > (c_temp - 2) and refined1_val[i, j] <= (c_temp - 1):
            G_B = refined1_val[i, j] - c_temp + 2
        elif refined1_val[i, j] > (c_temp - 1) and refined1_val[i, j] <= c_temp:
            G_B = c_temp - refined1_val[i, j]

        temp_fuzz.extend([G_B])
        c_temp += 1

        if refined1_val[i, j] <= (c_temp - 2) or refined1_val[i, j] > c_temp:
            G_AV = 0
        elif refined1_val[i, j] > (c_temp - 2) and refined1_val[i, j] <= (c_temp - 1):
            G_AV = refined1_val[i, j] - c_temp + 2
        elif refined1_val[i, j] > (c_temp - 1) and refined1_val[i, j] <= c_temp:
            G_AV = c_temp - refined1_val[i, j]

        temp_fuzz.extend([G_AV])
        c_temp += 1

        if refined1_val[i, j] <= (c_temp - 2) or refined1_val[i, j] > c_temp:
            G_G = 0
        elif refined1_val[i, j] > (c_temp - 2) and refined1_val[i, j] <= (c_temp - 1):
            G_G = refined1_val[i, j] - c_temp + 2
        elif refined1_val[i, j] > (c_temp - 1) and refined1_val[i, j] <= c_temp:
            G_G = c_temp - refined1_val[i, j]

        temp_fuzz.extend([G_G])
        c_temp += 1

        if refined1_val[i, j] <= (c_temp - 2) or refined1_val[i, j] > c_temp:
            G_VG = 0
        elif refined1_val[i, j] > (c_temp - 2) and refined1_val[i, j] <= (c_temp - 1):
            G_VG = refined1_val[i, j] - c_temp + 2
        elif refined1_val[i, j] > (c_temp - 1) and refined1_val[i, j] <= c_temp:
            G_VG = c_temp - refined1_val[i, j]

        temp_fuzz.extend([G_VG])
        c_temp += 1

        if refined1_val[i, j] <= 4:
            G_E = 0
        elif refined1_val[i, j] > 4 and refined1_val[i, j] <= 5:
            G_E = refined1_val[i, j] - 4

        temp_fuzz.extend([G_E])
    if refined1_val[i, no_genre + 1] <= 20:
        A_Y = 1
    elif refined1_val[i, no_genre + 1] <= 35 and refined1_val[i, no_genre + 1] > 20:
        A_Y = (35 - refined1_val[i, no_genre + 1]) / 15
    else:
        A_Y = 0
    temp_fuzz.extend([A_Y])

    if refined1_val[i, no_genre + 1] <= 20 or refined1_val[i, no_genre + 1] > 60:
        A_M = 0
    elif refined1_val[i, no_genre + 1] <= 35 and refined1_val[i, no_genre + 1] > 20:
        A_M = (refined1_val[i, no_genre + 1] - 20) / 15
    elif refined1_val[i, no_genre + 1] <= 45 and refined1_val[i, no_genre + 1] > 35:
        A_M = 1
    else:
        A_M = (60 - refined1_val[i, no_genre + 1]) / 15
    temp_fuzz.extend([A_M])

    if refined1_val[i, no_genre + 1] <= 45:
        A_o = 0
    elif refined1_val[i, no_genre + 1] <= 60 and refined1_val[i, no_genre + 1] > 45:
        A_o = (refined1_val[i, no_genre + 1] - 45) / 15
    else:
        A_o = 1
    temp_fuzz.extend([A_o])
    temp_fuzz.extend([refined1_val[i, no_genre + 2], refined1_val[i, no_genre + 3]])
    fuzzy_vals[i] = temp_fuzz
    # TEST:print fuzzy_vals[i]
    print("User", i, "processed!!\n")

fuzzy_refined = pd.DataFrame(fuzzy_vals, columns=columns_fuzzy)

print("Data Successsfully Fuzzyfied!")
print("************************Fuzzified Data**************************************")
print(fuzzy_refined.head(10))

# Final Step: Calculate MAE for 5 Folds
n_folds = 5  # Can change it acc. to your analysis
mae_folds = pd.DataFrame(columns=['Fold', 'MAE'])  # Stores 5 Fold MAE values

rating_values = result2[['user_id', 'item_id', 'rating']].values
gim_values = refined1.values
fuzz_gim = fuzzy_refined.values
mean_rating = []

# Adding the mean rating of each user in the 'refined1' dataframe
for i in user_vals:
    temp = grouped.get_group(i)
    mean_rating.extend([temp.rating.mean()])

refined1['mean_rating'] = mean_rating
rated_items = pd.DataFrame(columns=['user_id', 'item_list'])  # List of movies rated by every user

# Adding list of items rated by each user in a separate Data Frame(rated_items)
for i in user_vals:
    r = {'user_id': i}
    temp = grouped.get_group(i)
    r['item_list'] = temp.item_id.values
    rated_items = rated_items.append(r, ignore_index=True)

# For each folds
entry_mae_folds = dict()  # For each entry in the MAE_Folds DF(DataFrame)

print("Final Process....")
for fold in range(1, n_folds + 1, 1):
    print("Processing Fold", fold, ".....")
    start = 50 * (fold - 1)
    end = 50 * fold
    user_space = pd.DataFrame(columns=['user_id', 'MAE'])
    entry_mae_folds['Fold'] = fold
    user_vals_ndarray = np.array(user_vals)
    user_set = gim_values[start:end, 0]  # Test Users
    other_set = np.setdiff1d(user_vals_ndarray, user_set)
    # TEST: print user_set,other_set
    for i in user_set:
        print("Processing User", i, "....")
        entry_user_space = dict()
        entry_user_space['user_id'] = i
        neighbourhood = pd.DataFrame(columns=['user_id', 'n_id', 'distance'])  # neighbours of each user
        movies = pd.DataFrame(columns=['item_id', 'actual', 'predicted'])  # For each User: For each Movie

        for j in other_set:
            # TEST: print i,j
            entry_neighbourhood = dict()  # Everytime a new Dictionary
            entry_neighbourhood['user_id'] = i
            entry_neighbourhood['n_id'] = j  # Each new entry for neighbourhood dataframe
            a_fuzz = fuzzy_refined[fuzzy_refined.user_id == i][fuzzy_refined.columns[1:]].values
            b_fuzz = fuzzy_refined[fuzzy_refined.user_id == j][fuzzy_refined.columns[1:]].values
            d_fuzz = euclidean(a_fuzz, b_fuzz)
            a_eu = refined1[refined1.user_id == i][refined1.columns[1:6]].values
            b_eu = refined1[refined1.user_id == j][refined1.columns[1:6]].values
            d_eu = euclidean(a_eu, b_eu)
            entry_neighbourhood['distance'] = d_eu * d_fuzz
            neighbourhood = neighbourhood.append(entry_neighbourhood, ignore_index=True)
        neighbourhood = neighbourhood.sort(['distance'])
        neighbourhood = neighbourhood.head(15)
        neighbours = neighbourhood.n_id.values
        test = rated_items[rated_items.user_id == i].item_list.values
        test = test[0:(len(test) / 5) + 1][0]
        # TEST: print test
        for j in test.tolist():
            entry_movies = dict()
            entry_movies['item_id'] = j
            rated_members = []

            for k in neighbours:
                temp = rated_items[rated_items.user_id == k].item_list.values[0]
                if sum(temp == j) > 0:
                    rated_members.extend([k])
                    # TEST: print rated_members
            entry_movies['actual'] = result2[(result2.user_id == i) & (result2.item_id == j)].rating.values[0]
            k = 1.0 / (neighbourhood.distance.sum())
        if len(rated_members) == 0:
            entry_movies['predicted'] = np.nan
        else:
            pr = predict(refined1, i, k, neighbourhood, rated_members, result2, j)
            entry_movies['predicted'] = pr
        movies = movies.append(entry_movies, ignore_index=True)

    entry_user_space['MAE'] = calculate_mae(movies)
    user_space = user_space.append(entry_user_space, ignore_index=True)
    print("Processed User ", i, " !!")
entry_mae_folds['MAE'] = user_space.MAE.mean()
mae_folds = mae_folds.append(entry_mae_folds, ignore_index=True)
print("Processed Fold: ", fold, "!!")

print("\n\nEvery fold Processed!!\n\n\n")

print("************Final Results**************")
print(mae_folds)
