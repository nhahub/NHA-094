from google import genai

client = genai.Client(api_key="AIzaSyAPwT7N716JovCHACF8D-mhemIZU_odxxU")

response = client.models.generate_content(
    model="gemini-2.5-flash", contents="Explain how AI works in a few words"
)
print(response.text)
