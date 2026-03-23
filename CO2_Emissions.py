import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import time

# Đọc dữ liệu từ file CSV
data = pd.read_csv("CO2 Emissions.csv")

# Hiển thị các thông tin tổng quát về dữ liệu
# Kiểm tra số Hàng và Cột
print("Kiểm tra số hàng và cột: ")
print(data.shape , end='\n\n')
# In 20 hàng đầu tiên của tập dữ liệu
print("In 20 hàng đầu tiên của dữ liệu: ")
print(data.head(20) , end='\n\n')
# Kiểm tra loại dữ liệu của từng cột
print("Kiểm tra loại dữ liệu của từng cột: ")
print(data.info() , end='\n\n')
# Kiểm tra trùng lặp
print("Kiểm tra trùng lặp: ")
print(data.nunique() , end='\n\n')

# Đồ thị
# 1. Đồ thị hiển thị các nhà sản xuất xe hàng đầu chiếm tổng cộng 50% số xe trong tập dữ liệu.
# Đếm số lượng xe của từng nhà sản xuất
manufacturer = data['Make'].value_counts()

# Tính ngưỡng cho 50% số xe
fifty_percent = data.shape[0] / 2

# Xác định các nhà sản xuất hàng đầu chiếm 50% số xe
total = 0
count = 0
for count, val in enumerate(manufacturer.values):
    total += val
    if total > fifty_percent:
        count += 1  # Để bao gồm nhà sản xuất cuối cùng
        break

# In số lượng nhà sản xuất
print(f"Số lượng nhà sản xuất chiếm 50% số xe: {count}" , end='\n\n')

# Vẽ biểu đồ
plt.figure(figsize=(15, 6))
plt.bar(x=manufacturer.index[:count], height=manufacturer.values[:count], color='indigo')

# Thêm nhãn số lượng xe lên trên các cột
for index, value in enumerate(manufacturer.values[:count]):
    plt.text(index, value + 8, str(value), ha='center')

# Thêm lưới, tiêu đề, và nhãn
plt.grid(alpha=0.4)
plt.title('Các nhà sản xuất xe hàng đầu', fontsize=15)
plt.xlabel('Nhà sản xuất', fontsize=12)
plt.ylabel('Số lượng xe', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()

# Hiển thị biểu đồ
plt.show()

# 2. Trong số các nhà sản xuất xe hàng đầu, đồ thị hiển thị nhà sản xuất có mức tiết kiệm nhiên liệu trung bình cao nhất và thấp nhất, được đo bằng mức tiêu thụ nhiên liệu kết hợp (L/100km)
topManufacturer = manufacturer.index[:count]
fil_data = data[data['Make'].isin(topManufacturer)]

plt.figure(figsize=(15,5))
sns.boxplot(data=fil_data, x=fil_data['Make'], y=fil_data['Fuel Consumption Comb (L/100 km)'], palette='magma')
plt.xlabel('Nhà sản xuất')
plt.ylabel('Hiệu suất nhiên liệu')
plt.grid(alpha=0.4)
plt.title('So sánh hiệu suất nhiên liệu của các nhà sản xuất hàng đầu', fontsize=15)
plt.show()

# 3. Trong số các nhà sản xuất hàng đầu, đồ thị hiển thị nhà sản xuất có lượng khí thải CO2 trung bình cao nhất và thấp nhất (g/km)
plt.figure(figsize=(15,5))
sns.boxplot(data=fil_data, x=fil_data['Make'], y=fil_data['CO2 Emissions(g/km)'], palette='magma')
plt.xlabel('Nhà sản xuất')
plt.ylabel('Lượng khí thải CO2 (g/km)')
plt.grid(alpha=0.4)
plt.title('So sánh lượng khí thải CO2 của phương tiện (g/km) giữa các nhà sản xuất hàng đầu', fontsize=15)
plt.show()

# 4. Đồ thị thể hiện sự phân bố của các loại phương tiện trong tập dữ liệu
vehicles = data['Vehicle Class'].value_counts().sort_values()

# visualization
plt.figure(figsize=(15,8))
plt.barh(y=vehicles.index, width=vehicles.values, color='indigo')
plt.xlabel('Tần suất')

for index, values in enumerate(vehicles):
    plt.text(values+5, index, str(values), va='center')

plt.grid(alpha=0.4)
plt.title('Phân bố của các loại phương tiện', fontsize=15)
plt.show()

# 5. Đồ thị thể hiện loại phương tiện có hiệu suất nhiên liệu trung bình cao nhất và thấp nhất, được đo bằng mức tiêu thụ nhiên liệu kết hợp (L/100km)
plt.figure(figsize=(15,5))
sns.boxplot(data=data, x=data['Vehicle Class'], y=data['Fuel Consumption Comb (L/100 km)'], palette='magma')
plt.xlabel('Loại Xe')
plt.xticks(rotation=90)
plt.ylabel('Hiệu suất nhiên liệu')
plt.grid(alpha=0.4)
plt.title('So sánh hiệu suất nhiên liệu giữa các loại phương tiện khác nhau', fontsize=15)
plt.xticks(rotation=45)
plt.show()

# 6. Đồ thị thể hiện loại xe có lượng khí thải CO2 trung bình cao nhất và thấp nhất (g/km)
plt.figure(figsize=(15,5))
sns.boxplot(data=data, x=data['Vehicle Class'], y=data['CO2 Emissions(g/km)'], palette='magma')
plt.xlabel('Loại Xe')
plt.xticks(rotation=90)
plt.ylabel('Khí Thải CO2 (g/km)')
plt.grid(alpha=0.4)
plt.title('So sánh lượng khí Thải CO2 (g/km) giữa các loại xe khác nhau', fontsize=15)
plt.xticks(rotation=45)
plt.show()

# 7. Đồ thị thể hiện tỷ lệ giữa loại hộp số tự động và hộp số thủ công trong từng loại xe khác nhau.
# Thêm 1 cột mới
data['Transmission_Type'] = data['Transmission'].apply(lambda x: 'Tự động' if x.startswith('A') else 'Số sàn')

plt.figure(figsize=(15,6))
ax=sns.countplot(data=data, x=data['Vehicle Class'], hue=data['Transmission_Type'], palette='magma')
plt.xticks(rotation=90)
plt.xlabel(' ')
plt.ylabel('Tần suất')
plt.grid(alpha=0.4)
plt.title('Dạng truyền động trong các loại xe khác nhau', fontsize=15)
plt.xticks(rotation=45)
plt.show()

# 8. Đồ thị thể hiện loại nhiên liệu phổ biến nhất được các phương tiện sử dụng
# Thay thế giá trị trong cột 'Fuel Type'
data['Fuel Type'] = data['Fuel Type'].replace(['X', 'Z', 'E', 'D', 'N'], ['Xăng thường','Xăng cao cấp','Etanol','Dầu diesel','Khí tự nhiên'])

fuel_type = data['Fuel Type'].value_counts().sort_values()

plt.figure(figsize=(12,5))
plt.barh(y=fuel_type.index, width=fuel_type.values, color='indigo')
plt.xlabel('Tần suất')

for index, values in enumerate(fuel_type):
    plt.text(values+20, index, str(values), va='center')

plt.grid(alpha=0.4)
plt.title('Loại nhiên liệu phổ biến nhất sử dụng bởi phương tiện', fontsize=15)
plt.show()

# 9. Đồ thị thể hiện sự khác biệt về hiệu suất nhiên liệu và lượng khí thải CO2 giữa các loại xe sử dụng nhiên liệu khác nhau
data_loc = data[data['Fuel Type']!='Natural Gas']
labels = ['Xăng Thường','Xăng Cao Cấp','Ethanolo','Dầu Diesel']

figure, axes = plt.subplots(2,1, figsize=(10,10))
sns.boxplot(data=data_loc, x=data_loc['Fuel Type'], y=data_loc['CO2 Emissions(g/km)'], palette='magma', ax=axes[0])
axes[0].grid(alpha=0.4)
axes[0].set_xticklabels(labels)
axes[0].set_title('Khí Thải CO2 (g/km)', fontsize=12)
sns.boxplot(data=data_loc, x=data_loc['Fuel Type'], y=data_loc['Fuel Consumption Comb (L/100 km)'], palette='magma', ax=axes[1])
axes[1].grid(alpha=0.4)
axes[1].set_xticklabels(labels[:4])
axes[1].set_title('Tiêu Thụ Nhiên Liệu (L/100km)', fontsize=12)
figure.suptitle('So sánh giữa các loại xe sử dụng nhiên liệu khác nhau', fontsize=15)
plt.tight_layout(pad=1)
plt.show()

# 10. Đồ thị thể hiện sự tương quan giữa kích thước động cơ và hiệu suất nhiên liệu
plt.figure(figsize=(15,5))
ax=sns.relplot(data=data, x=data['Engine Size(L)'], y=data['Fuel Consumption Comb (L/100 km)'], 
               hue=data['Transmission_Type'], palette='magma', height=6, aspect=1.5)
plt.title('Kích thước động cơ so với hiệu suất nhiên liệu (trong Thành phố và Đường cao tốc)', fontsize=15)
plt.show()

# 11. Các đồ thị thể hiện mối quan hệ giữa lượng khí thải CO2 và tiêu thụ nhiên liệu (trong thành phố, trên đường cao tốc, hoặc tổng hợp) giữa các xe có hộp số tự động và hộp số sàn
plt.figure(figsize=(15,5))
sns.relplot(data=data, x=data['CO2 Emissions(g/km)'], y=data['Fuel Consumption City (L/100 km)'],
hue=data['Transmission_Type'], palette='magma', height=6, aspect=1.5)
plt.title('Lượng khí thải CO2 (g/km) so với hiệu suất nhiên liệu (trong thành phố) của các xe có hộp số tự động & hộp số sàn', fontsize=15)
plt.show()

plt.figure(figsize=(15,5))
sns.relplot(data=data, x=data['CO2 Emissions(g/km)'], y=data['Fuel Consumption Hwy (L/100 km)'],
hue=data['Transmission_Type'], palette='magma', height=6, aspect=1.5)
plt.title('Lượng khí thải CO2 (g/km) so với hiệu suất nhiên liệu (trên đường cao tốc) của các xe có hộp số tự động & hộp số sàn', fontsize=15)
plt.show()

plt.figure(figsize=(15,5))
sns.relplot(data=data, x=data['CO2 Emissions(g/km)'], y=data['Fuel Consumption Comb (L/100 km)'],
hue=data['Transmission_Type'], palette='magma', height=6, aspect=1.5)
plt.title('Lượng khí thải CO2 (g/km) so với hiệu suất nhiên liệu (trong thành phố & trên đường cao tốc) của các xe có hộp số tự động & hộp số sàn', fontsize=15)
plt.show()

# 12. Các đồ thị thể hiện mối quan hệ giữa lượng khí thải CO2 và tiêu thụ nhiên liệu (trong thành phố, trên đường cao tốc, hoặc tổng hợp) giữa các loại xe sử dụng nhiên liệu khác nhau
plt.figure(figsize=(15,5))
ax=sns.relplot(data=data, x=data['CO2 Emissions(g/km)'], y=data['Fuel Consumption City (L/100 km)'],
hue=data['Fuel Type'], palette='magma', height=6, aspect=1.5)
plt.title('Lượng khí thải CO2 (g/km) so với hiệu suất nhiên liệu (trong thành phố) của các loại xe sử dụng nhiên liệu khác nhau', fontsize=15)
plt.show()

plt.figure(figsize=(15,5))
ax=sns.relplot(data=data, x=data['CO2 Emissions(g/km)'], y=data['Fuel Consumption Hwy (L/100 km)'],
hue=data['Fuel Type'], palette='magma', height=6, aspect=1.5)
plt.title('Lượng khí thải CO2 (g/km) so với hiệu suất nhiên liệu (trên đường cao tốc) của các loại xe sử dụng nhiên liệu khác nhau', fontsize=15)
plt.show()

plt.figure(figsize=(15,5))
ax=sns.relplot(data=data, x=data['CO2 Emissions(g/km)'], y=data['Fuel Consumption Comb (L/100 km)'],
hue=data['Fuel Type'], palette='magma', height=6, aspect=1.5)
plt.title('Lượng khí thải CO2 (g/km) so với hiệu suất nhiên liệu (trong thành phố & trên đường cao tốc) của các loại xe sử dụng nhiên liệu khác nhau', fontsize=15)
plt.show()

# 13. Đồ thị thể hiện mối tương quan giữa kích thước động cơ, số xy-lanh, tiêu thụ nhiên liệu (trong thành phố, trên đường cao tốc, trong thành phố và trên đường cao tốc) và lượng khí thải CO2.
# Tính toán hệ số tương quan Pearson giữa các biến
num_vars = ['Engine Size(L)', 'Cylinders','Fuel Consumption City (L/100 km)','Fuel Consumption Hwy (L/100 km)', 'Fuel Consumption Comb (L/100 km)', 'Fuel Consumption Comb (mpg)', 'CO2 Emissions(g/km)']
correlation = data[num_vars].corr()

# Trực quan hóa mối tương quan trong heatmap
plt.figure(figsize=(10,8))
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.xticks(rotation=45)
plt.show()


# Huấn luyện và đánh giá các mô hình
print(end='\n\n')
print("Huấn luyện và đánh giá mô hình")

# Chia dữ liệu cho mô hình hồi quy
features = ['Engine Size(L)', 'Cylinders', 'Fuel Consumption City (L/100 km)', 'Fuel Consumption Hwy (L/100 km)', 'Fuel Consumption Comb (L/100 km)', 'Fuel Consumption Comb (mpg)']
X_reg = data[features]
y_reg = data['CO2 Emissions(g/km)']
# Chia tập dữ liệu thành training và testing cho mô hình hồi quy
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# # Chọn tất cả cột
# features = ['Make', 'Model', 'Vehicle Class', 'Engine Size(L)', 'Cylinders', 'Transmission', 'Fuel Type', 'Fuel Consumption City (L/100 km)', 'Fuel Consumption Hwy (L/100 km)', 'Fuel Consumption Comb (L/100 km)', 'Fuel Consumption Comb (mpg)']
# # Biến đổi các cột phân loại thành biến số sử dụng OneHotEncoder
# df_encoded = pd.get_dummies(data[features], drop_first=True)
# y_reg = data['CO2 Emissions(g/km)']
# # Chia tập dữ liệu thành training và testing cho mô hình hồi quy
# X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(df_encoded, y_reg, test_size=0.2, random_state=42)

# Huấn luyện và đánh giá mô hình Linear Regression
start_time = time.time()
model_lr = LinearRegression()
model_lr.fit(X_train_reg, y_train_reg)
y_pred_lr = model_lr.predict(X_test_reg)
end_time = time.time()

r2_lr = r2_score(y_test_reg, y_pred_lr)
mae_lr = mean_absolute_error(y_test_reg, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test_reg, y_pred_lr))

# Hiển thị đánh giá các thông số của mô hình Linear Regression
print("Đánh giá các thông số của Linear Regression:")
print(f"R2 Score: {r2_lr}")
print(f"Mean Absolute Error: {mae_lr}")
print(f"Root Mean Squared Error: {rmse_lr}")
print(f"Thời gian thực hiện: {end_time - start_time} giây")
# Hiển thị thông số của mô hình Linear Regression
print("Các tham số của Linear Regression:")
print(model_lr.get_params(), end='\n\n')

# Huấn luyện và đánh giá mô hình Random Forest
start_time = time.time()
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train_reg, y_train_reg)
# Dự đoán trên tập test
y_pred_rf = model_rf.predict(X_test_reg)
end_time = time.time()

# Đánh giá hiệu suất của mô hình Random Forest Regressor
r2_rf = r2_score(y_test_reg, y_pred_rf)
mae_rf = mean_absolute_error(y_test_reg, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test_reg, y_pred_rf))

# Hiển thị đánh giá các thông số của mô hình Random Forest
print("Đánh giá các thông số của Random Forest:")
print(f"R2 Score: {r2_rf}")
print(f"Mean Absolute Error: {mae_rf}")
print(f"Root Mean Squared Error: {rmse_rf}")
print(f"Thời gian thực hiện: {end_time - start_time} giây")
# Hiển thị thông số của mô hình Random Forest
print("Các tham số của Random Forest:")
print(model_rf.get_params(), end='\n\n')

# Huấn luyện và đánh giá mô hình Decision Tree Regressor
start_time = time.time()
model_dtr = DecisionTreeRegressor(random_state=42)
model_dtr.fit(X_train_reg, y_train_reg)
# Dự đoán trên tập test
y_pred = model_dtr.predict(X_test_reg)
end_time = time.time()

# Đánh giá hiệu suất của mô hình
r2 = r2_score(y_test_reg, y_pred)
mae = mean_absolute_error(y_test_reg, y_pred)
# rmse = mean_squared_error(y_test_reg, y_pred, squared=False)
rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred))

print("Đánh giá các thông số của Decision Tree Regressor:")
print(f"R2 Score: {r2}")
print(f"Mean Absolute Error: {mae}")
print(f"Root Mean Squared Error: {rmse}")
print(f"Thời gian thực hiện: {end_time - start_time} giây")
# Hiển thị thông số của mô hình Decision Tree Regressor
print("Các tham số của Decision Tree Regressor:")
print(model_dtr.get_params(), end='\n\n')

# Vẽ biểu đồ scatter plot cho kết quả thực tế vs dự đoán cho Linear Regression, Random Forest và Decision Tree Classification
plt.figure(figsize=(18, 6))

# Linear Regression
plt.subplot(1, 3, 1)
plt.scatter(y_test_reg, y_pred_lr, color='blue', alpha=0.5)
plt.plot([min(y_test_reg), max(y_test_reg)], [min(y_test_reg), max(y_test_reg)], color='red')
plt.title('Linear Regression: Thực tế vs Dự đoán')
plt.xlabel('Giá trị thực tế')
plt.ylabel('Giá trị dự đoán')

# Random Forest
plt.subplot(1, 3, 2)
plt.scatter(y_test_reg, y_pred_rf, color='green', alpha=0.5)
plt.plot([min(y_test_reg), max(y_test_reg)], [min(y_test_reg), max(y_test_reg)], color='red')
plt.title('Random Forest: Thực tế vs Dự đoán')
plt.xlabel('Giá trị thực tế')
plt.ylabel('Giá trị dự đoán')

# Decision Tree
plt.subplot(1, 3, 3)
plt.scatter(y_test_reg, y_pred, color='orange', alpha=0.5)
plt.plot([min(y_test_reg), max(y_test_reg)], [min(y_test_reg), max(y_test_reg)], color='red')
plt.title('Decision Tree: Thực tế vs Dự đoán')
plt.xlabel('Giá trị thực tế')
plt.ylabel('Giá trị dự đoán')

plt.tight_layout()
plt.show()

# Xuất kết quả dự đoán tốt nhất (Random Forest Regressor) ra file CSV
results_df = pd.DataFrame({'Actual': y_test_reg, 'Predicted': y_pred_rf})
results_df.to_csv('random_forest_regressor_predictions.csv', index=False)

print("Kết quả dự đoán đã được xuất ra file random_forest_regressor_predictions.csv")