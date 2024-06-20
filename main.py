import streamlit as st
import tensorflow as tf
import numpy as np
def model_pridiction(test_image):
    model = tf.keras.models.load_model('trained_model.keras')
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    prediction = model.predict(np.expand_dims(input_arr, axis=0))
    result_index = np.argmax(prediction)       #max index of the matrix of this prediction (mathing 8.4312719)
    return result_index

#Side bar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition"])

#Home page
if(app_mode=="Home"):
    st.header("Plant Leaf Disease Detection System")
    st.image("homePage.jpg")
    st.write("This App is developed by the help of CNN(Convloution Neural Networks).")
elif(app_mode=="About"):
    st.header("About")
    st.write("This website is free and this may not give the accurate every time the result may vary due to clarity of images and also due to the fact that the model prediction may not be correct every time.")
    st.write("This website only suggest you the probable disease.")
    st.write("All the copy rights are with the B12 team.")
    st.image("SDMCET.jpg")
    st.write("SDM College of Engineering and technology Dharwad")
    st.write("Department of Computer Science Engineering")
    st.write("ADITYA ANAND JOSHI")



elif(app_mode=="Disease Recognition"):
    st.header("Leaf Disease Recognition")
    test_image=st.file_uploader("Choose an image:")
    if(st.button("Show image")):
        st.image(test_image,use_column_width=True)
    if(st.button("Predict")):
        st.write("Our prediction")
        result_index=model_pridiction(test_image)
        class_name = ['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Blueberry___healthy',
 'Cherry_(including_sour)___Powdery_mildew',
 'Cherry_(including_sour)___healthy',
 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn_(maize)___Common_rust_',
 'Corn_(maize)___Northern_Leaf_Blight',
 'Corn_(maize)___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Raspberry___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']
         class_des = [
                'Fungal disease caused by the ascomycete fungus Venturia inaequalis',
                'Common fungal disease that affects apple trees, caused by the fungus Botryosphaeria obtusa',
                'Fungal disease caused by Gymnosporangium juniperi-virginianae',
                'No disease found, the plant is healthy',
                'No disease found, the plant is healthy', 
                'Powdery mildew of sweet and sour cherry is caused by Podosphaera clandestina, an obligate biotrophic fungus',
                'No disease found, the plant is healthy',
                'Cercospora leaf spot and gray leaf spot are two common fungal diseases that affect corn (maize) crops',
                'Common rust is a foliar disease of corn favored by moist and cool conditions',
                'Northern Corn Leaf Blight (NCLB) is a foliar disease of corn (maize) caused by the fungus Exserohilum turcicum, also known as Setosphaeria turcica',
                'No disease found, the plant is healthy',
                'Fungal disease caused by the ascomycetous fungus Guignardia bidwellii',
                'Esca, also known as Black Measles, is a fungal disease that affects grapevines',
                'Isariopsis is a genus of fungi in the family Mycosphaerellaceae',
                'No disease found, the plant is healthy',
                'Citrus Greening, also known as Huanglongbing (HLB), is considered to be the most destructive disease of citrus',
                'Other names for the disease are bacteriosis, bacterial leaf spot, or bacterial shot hole',
                'No disease found, the plant is healthy',
                'Bacterial leaf spot, caused by Xanthomonas campestris pv. vesicatoria',
                'No disease found, the plant is healthy',
                'Early blight is a fungal disease caused by the pathogen Alternaria solani that affects potatoes and tomatoes',
                'Potato late blight is a serious disease caused by the oomycete pathogen Phytophthora infestans',
                'No disease found, the plant is healthy',
                'No disease found, the plant is healthy',
                'No disease found, the plant is healthy',
                'Blotchy patches of white-gray powdery spots that are dry to the touch on the surface of leaves',
                'Strawberry leaf scorch is a fungal disease caused by Diplocarpon earlianum, which affects strawberry plants',
                'No disease found, the plant is healthy',
                'Bacterial spot can occur wherever tomatoes are grown, but is found most frequently in warm, wet climates, as well as in greenhouses',
                'Tomato early blight is a common fungal disease that affects tomato plants, caused by the fungus Alternaria linariae',
                'Tomato Late Blight is a fungal disease that affects tomatoes and potatoes',
                'Tomato leaf mold is a fungal disease that affects the foliage of tomatoes, particularly those grown in greenhouses and high tunnels',
                'Septoria leaf spot is a common disease that affects tomato plants, causing small dark spots to appear on the leaves',
                'Tomato spider mites and two-spotted spider mites are tiny, eight-legged arachnids that feed on the sap of plants, causing damage and discoloration',
                'Tomato target spot is a fungal disease that affects tomato plants, causing circular to irregular-shaped lesions on the leaves, stems, and fruit. The disease is caused by the fungus Corynespora cassiicola',
                'DNA virus from the genus Begomovirus and the family Geminiviridae',
                'The virus causes mosaic patterns on leaves, leaf distortions, stunted growth, and bronzing and marbling patterns on the fruit',
                'No disease found, the plant is healthy'
            ]
        st.success("Model is predicting it's a {} -> {}".format(class_name[result_index], class_des[result_index]))
