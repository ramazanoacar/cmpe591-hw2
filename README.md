# DQN Training on Hw2Env

Bu proje, **Deep Q-Network (DQN)** algoritmasını kullanarak **Hw2Env** ortamında bir ajan eğitmek için oluşturulmuştur.

## Eğitim Süreci

Eğitim sürecinde aşağıdaki parametreleri kullandık:

```python
num_episodes = 3800
update_frequency = 10
target_update_frequency = 200
epsilon = 1.0
epsilon_decay = 0.997
epsilon_min = 0.03
batch_size = 64
gamma = 0.99
buffer_size = 100000
learning_rate = 1e-3
```

Eğitimi max_timestep 100 ile gerçekleştirdik. Yiğit hocanın parametreleriyle yapılan uzun süreli bir eğitimin ardından, datayı kaydetmediğimizi fark ettik ve çıktılar boştu. Bu süreçten öğrendiklerimizi kullanarak epsilon decay ve min epsilon değerlerini yukarıda belirttiğimiz şekilde güncelledik.



![DQN Training Plot](/dqn_training_plot.png)

1️⃣ Genel Eğitim Grafiği

Bu grafik, eğitim sürecinde ajanımızın ödül kazanma performansını göstermektedir. Eğitim ilerledikçe ortalama ödül değerinin arttığı gözlemlenmiştir.



2️⃣ Ödül Grafiği
![Reward](/reward.png)
Her bir episode için ajanımızın aldığı toplam ödül değerleri gösterilmektedir.

3️⃣ Adım Başına Ödül (Reward Per Step)
![RPS](/rps.png)
Bu grafik, her adım başına alınan ortalama ödülleri gösterir. Öğrenme sürecinin istikrarlı hale gelmesiyle ödüller artış eğilimi göstermektedir.

4️⃣ Smoothed Reward (Pencere Boyutu: 50)
![Smoothed Reward 50](/smoothed_reward_50.png)
Bu üstteki grafik, 50 pencere genişliğinde hareketli ortalama alınarak ödüllerin nasıl değiştiğini gösterir. Eğitim süreci boyunca daha düzenli bir artış eğilimi gözlemlenmektedir.

5️⃣ Smoothed Reward (Pencere Boyutu: 100)
![Smoothed Reward 100](/smoothed_reward_100.png)
Bu üstteki grafik, 100 pencere genişliğinde hareketli ortalama ile daha düzgün hale getirilmiş ödülleri göstermektedir. Eğitim ilerledikçe ortalama ödüllerde yükselme olduğu açıkça görülmektedir.

6️⃣ Smoothed Reward Per Step (Pencere Boyutu: 50)
![Smoothed RPS 50](/smoothed_rps_50.png)
Bu üstteki grafik, 50 pencere genişliğinde hareketli ortalama ile adım başına alınan ödüllerin değişimini gösterir. Eğitim sürecinde kararlı bir artış gözlemlenmiştir.

7️⃣ Smoothed Reward Per Step (Pencere Boyutu: 100)
![Smoothed RPS 100](/smoothed_rps_100.png)
Daha büyük bir pencere genişliği ile adım başına ödüllerdeki genel eğilim daha net bir şekilde görülmektedir. Ödüllerde zamanla artış olduğu gözlemlenmiştir.







