# cv-project
# Computer Vision Project
Elbrus Bootcamp | Phase-2 | Team Project 
## Team
* [Daniil Lvov](https://github.com/Norgan97)
* [Dmitry Budazhapov](https://github.com/DmitryDorzhievich)
* [Larisa Khlapushina](https://github.com/Khlapushina)
___
This repository contains a multipage application project using Streamlit. The project was worked on by: *Lvov Daniil, Budazhapov Dmitry, Khlapushina Larisa**
___
## Tasks 
This application solves several tasks:
1. Object detection using YOLOv5.
   1. Intended for brain tumor detection from a photograph.
![](https://cdn.readovka.ru/n/1104148/1200x630/ec7da68ba3.jpg)
   2. For detecting Spiderman and Ironman. You can upload your photo, and the application will highlight the tumor area in the image or show the comic book heroes.
![](https://s2.best-wallpaper.net/wallpaper/2560x1440/1906/Iron-Man-and-Spider-man-DC-comics_2560x1440.jpg)
___
2. Object detection using YOLOv8:
   - Used for detecting wind turbines and cable towers. You can upload a photo, and our application will highlight all the wind turbine images in your picture. Additionally, you can watch a video test in the application where our model will also highlight the locations of these objects.
![](https://get.pxhere.com/photo/field-windmill-wind-cumulus-machine-wind-turbine-electricity-energy-england-power-mill-grassland-wind-farm-wind-turbines-835672.jpg)
___
3. Document denoising using an autoencoder:
   - This application allows you to get a clean image of the uploaded text photo with various defects."
3. Очищение документов от шумов с помощью автоэнкодера:
**Это приложение позволит вам получить чистое изображение загруженного вами фото текста с различными дефектами**
![](https://github.com/Norgan97/cv-project/blob/main/2.png)
![](https://github.com/Norgan97/cv-project/blob/main/21.png)
___
## Deployment
The service is implemented on [Streamlit](https://cv-project-as5vxgxapfhae8psceu5xf.streamlit.app)
_
## How to run locally?
## To run the provided applications on your computer, follow these steps:

1. Clone this repository to your local machine.
2. Install the required libraries by running the command *pip install -r requirements.txt* in your terminal or command prompt.
3. Once the libraries are installed, navigate to the repository's directory in your terminal.
4. Run the command *streamlit run main.py* in your terminal to start the application.

This will launch the Streamlit server, and you can access the applications by opening a browser window and navigating to the specified URL.
