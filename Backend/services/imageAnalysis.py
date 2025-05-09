from groq import Groq

client = Groq(api_key="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")

def analyze_medical_image(image_url: str) -> str:
    print("doing image analysis",image_url)
    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are a radiologist's assistant. Analyze the medical image provided below. "
                            "Provide a detailed analysis of what is shown, including any visible abnormalities, "
                            "possible diagnoses, areas of concern, and suggested next steps. "
                            "Return your response ONLY in this exact JSON format:\n\n"
                            "{\n"
                            "  \"findings\": \"string - A detailed description of what is seen in the image\",\n"
                            "  \"abnormalities\": [\"string - List of any abnormalities found, or an empty array\"],\n"
                            "  \"possible_diagnoses\": [\"string - Potential medical diagnoses based on the findings\"],\n"
                            "  \"areas_of_concern\": [\"string - Specific anatomical areas that need further review\"],\n"
                            "  \"recommendations\": [\"string - Suggested next steps such as follow-up tests, treatments, or specialist referrals\"]\n"
                            "}\n\n"
                            "Do not include any extra explanation, markdown, or text. Just return the JSON."
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    }
                ]
            }
        ],
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
        stream=False,
        stop=None,
    )
    print(response.choices[0].message.content)
    return response.choices[0].message.content
