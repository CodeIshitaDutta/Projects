import openai
import pandas as pd
import streamlit as st



# Setup Credentials
openai.api_key = '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'

# First Question Functions

def api_req(given_prompt):
    # Make the API request
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=given_prompt,
        max_tokens=1000  # Adjust the max_tokens value as per your requirements
    )
    # Extract the generated text from the response
    generated_text = response.choices[0].text.strip()
    return(generated_text)


def cond_search(condition):
    #condition = str(input("What condition would you like to search? "))

    prompt= """Methods:
    Ayurveda
    Homeopathy
    Conventional/Western Medicine
    Oriental Medicine (Traditional Chinese Medicine):
    Naturopathy
    Traditional Medicine

    Please provide the Method name, one common remedy, Potency, Dosage, Duration, Prescription required for {} using the above methods in a list seperated by newlines.""".format(condition)

    # Split the generated text into individual lines
    generated_text = api_req(prompt)
    lines = generated_text.split("\n\n")
    # Create lists to store the table data
    styles = []
    #illnesses = []
    #symptoms = []
    remedies = []
    potencies = []
    dosage = []
    duration = []
    prescription_required = []

    # Process each line and print the corresponding table row
    for i in range (len(lines)):
        line = lines[i]
        # Split the line into cells based on the pipe character
        cells = [cell.strip() for cell in line.split("\n")]
        # Extract the relevant cells
        remedy_style = cells[0].split(':')[0]
    #    illness = cells[1].split(': ')[1]
    #    symptom = cells[2].split(': ')[1]
        remedy = cells[1].split(': ')[1]
        potency = cells[2].split(': ')[1]
        dosage_value = cells[3].split(': ')[1]
        duration_value = cells[4].split(': ')[1]
        prescription = cells[5].split(': ')[1]
        
        # Append the data to the respective lists
        styles.append(remedy_style)
    #    illnesses.append(illness)
    #    symptoms.append(symptom)
        remedies.append(remedy)
        potencies.append(potency)
        dosage.append(dosage_value)
        duration.append(duration_value)
        prescription_required.append(prescription)
    
    data_frame_setup = {
        "Practice Type": styles,
    #    "Illness": illnesses,
    #    "Symptoms": symptoms,
        "Common Remedy": remedies,
        "Potency": potencies,
        "Dosage": dosage,
        "Duration": duration,
        "Prescription Required": prescription_required
    }

    # Create a dataframe from the dictionary
    retframe = pd.DataFrame(data_frame_setup)
    return retframe

def deepsearch(dataset, row, entry):
    rowval = dataset.iloc[row]
    prompt_2= """Please produce more information about {} in {} with {} potency, {} dosage for {} as a treatment for {} from credible sources""".format(rowval['Common Remedy'], rowval['Practice Type'], rowval['Potency'], rowval['Dosage'], rowval['Duration'], entry)

    # Make the API request
    response_2 = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt_2,
        max_tokens=1000  # Adjust the max_tokens value as per your requirements
    )
    # Extract the generated text from the response
    generated_text_2 = response_2.choices[0].text.strip()

    # Split the generated text into individual lines
    lines_2 = generated_text_2.split("\n\n")
    for line in lines_2:
        print(line, "\n")
    return()#(lines_2)




def home_page():
    st.title("InForMed")
    st.write("""Welcome to InForMed, a place where more than one type of treatment is explored.
    My goal is to have 6 different types of treatments for any diagnosis.""")
    user_input = st.text_input("Please enter your diagnosis below, and click on the \"Submit\" button ")
    # Make API request
    df = None
    if st.button("Submit"):
        st.write("""Your diagnosis has been submitted. As I am still under development, 
        please make sure the following output is a neat 6 row table. If it is not a table, please submit again.""")
        df = cond_search(user_input)
        # Display the table
        st.dataframe(df)
        
    return(df, user_input)
        
        # st.write("\n\nWhich Practice would you like to explore more?")
        
        # if st.button("Ayurveda"):
        #     deepsearch(df, 0, user_input)
        # elif st.button("Homeopathy"):
        #     deepsearch(df, 1, user_input)
        # elif st.button("Conventional/Western Medicine"):
        #     deepsearch(df, 2, user_input)
        # elif st.button("Oriental Medicine (Traditional Chinese Medicine)"):
        #     deepsearch(df, 3, user_input)
        # elif st.button("Naturopathy"):
        #     deepsearch(df, 4, user_input)
        # elif st.button("Traditional Medicine"):
        #     deepsearch(df, 5, user_input)

def run_app():
    df, user_input = home_page()
    st.write("\n\nWhich Practice would you like to explore more?")
    
    if st.button("Ayurveda"):
        deepsearch(df, 0, user_input)
    elif st.button("Homeopathy"):
        deepsearch(df, 1, user_input)
    elif st.button("Conventional/Western Medicine"):
        deepsearch(df, 2, user_input)
    elif st.button("Oriental Medicine (Traditional Chinese Medicine)"):
        deepsearch(df, 3, user_input)
    elif st.button("Naturopathy"):
        deepsearch(df, 4, user_input)
    elif st.button("Traditional Medicine"):
        deepsearch(df, 5, user_input)


# Run the app
if __name__ == '__main__':
    run_app()