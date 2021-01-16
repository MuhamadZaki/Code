import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


# In[2]:


# Muat kumpulan data diabetes
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)


# In[3]:


# Hanya menggunakan satu fitur
diabetes_X = diabetes_X[:, np.newaxis, 2]


# In[4]:


# Membagi data menjadi kumpulan pelatihan/pengujian
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]


# In[5]:


# Membagi target menjadi set pelatihan/pengujian
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]


# In[6]:



# Membuat objek regresi linear
regr = linear_model.LinearRegression()


# In[7]:


# Latih model menggunakan set pelatihan
regr.fit(diabetes_X_train, diabetes_y_train)


# In[8]:


# Membuat prediksi menggunakan set pengujian
diabetes_y_pred = regr.predict(diabetes_X_test)


# In[9]:



# Koefisien
print('Coefficients: \n', regr.coef_)
# Kesalahan kuadrat rata-rata
print('Mean squared error: %.2f'
      % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# Koefisien penentuan: 1
print('Coefficient of determination: %.2f'
      % r2_score(diabetes_y_test, diabetes_y_pred))


# In[12]:



# Output plot
plt.scatter(diabetes_X_test, diabetes_y_test,  color='orange')
plt.plot(diabetes_X_test, diabetes_y_pred, color='black', linewidth=3)
plt.xticks(())
plt.yticks(())

plt.show()


# In[ ]: