import streamlit as st
from diffusers import StableDiffusionPipeline
from PIL import Image
import torch

# Load the fine-tuned or pre-trained Stable Diffusion model
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16).to("cpu")

# Streamlit interface
st.title("Interior Design Concept Generator")
st.write("Generate interior design concepts for homes or office spaces based on text prompts.")

# User input: design description
design_prompt = st.text_area("Describe the design you'd like to see", "A modern living room with white walls and wooden furniture")

# User input: design style
style = st.selectbox("Choose a design style", ["Modern", "Minimalist", "Industrial", "Scandinavian", "Bohemian", "Rustic", "Eclectic"])

# Generate Button
if st.button("Generate Design"):
    with st.spinner("Generating design..."):
        # Combine user inputs into a full prompt
        full_prompt = f"{style} style {design_prompt}"

        # Generate the image from the model
        generated_image = pipe(full_prompt).images[0]

        # Display the generated image in Streamlit
        st.image(generated_image, caption="Generated Interior Design", use_column_width=True)

        # Option to save the image locally
        if st.button("Save Design"):
            file_name = f"{style}_design.png"
            generated_image.save(file_name)
            st.success(f"Design saved as {file_name}")

# Export options for mood boards or blueprints
download_option = st.radio("Export Design as:", ["Mood Board", "Blueprint"])
if st.button("Export Design"):
    if download_option == "Mood Board":
        st.write("Mood Board export is coming soon...")
    elif download_option == "Blueprint":
        st.write("Blueprint export is coming soon...")
