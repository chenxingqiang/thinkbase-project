from typing import Optional
from pydantic import BaseModel, Field
from agentflow.ell.integration import ELLIntegration
from agentflow.ell.types.message import Message, MessageRole, MessageType

# Get singleton instance
ell = ELLIntegration()

# Initialize ELL
ell.configure({
    "enabled": True,
    "tracking_enabled": True,
    "store": "./logdir",
    "verbose": True,
    "autocommit": True,
    "model": "gpt-4",
    "default_model": "gpt-4",
    "temperature": 0.1,
    "max_tokens": 200,
    "metadata": {
        "type": "text",
        "format": "plain"
    }
})

class WeatherInfo(BaseModel):
    """Weather information for a location."""
    location: str = Field(description="The location to get weather for")
    temperature: float = Field(description="The temperature in Celsius")
    condition: str = Field(description="The weather condition (e.g., sunny, rainy)")

class TravelAdvice(BaseModel):
    """Travel advice for a destination."""
    destination: str = Field(description="The destination to get advice for")
    advice: str = Field(description="The travel advice")

@ell.with_ell(mode="simple")
async def get_weather(location: str) -> WeatherInfo:
    """Get the current weather for a given location."""
    # Simulated weather API call
    return WeatherInfo(
        location=location,
        temperature=25.0,
        condition="sunny"
    )

@ell.with_ell(mode="simple")
async def travel_planner(destination: str) -> TravelAdvice:
    """Plan a trip based on the destination and current weather."""
    try:
        # Get weather information
        weather = await get_weather(destination)
        
        # Generate travel advice based on weather
        advice = f"""Based on the current weather in {destination} (Temperature: {weather.temperature}°C, Condition: {weather.condition}), here are my recommendations:

1. What to Pack:
   - Light, breathable clothing
   - Comfortable walking shoes
   - Sunglasses and sunscreen
   - A light jacket for evenings

2. Best Activities:
   - Explore outdoor attractions and landmarks
   - Visit local cafes and outdoor restaurants
   - Take a walking tour of the city
   - Enjoy parks and gardens

3. Weather-related Precautions:
   - Stay hydrated throughout the day
   - Seek shade during peak sun hours
   - Apply sunscreen regularly
   - Consider indoor activities during the hottest part of the day"""
        
        return TravelAdvice(
            destination=destination,
            advice=advice
        )
        
    except Exception as e:
        return TravelAdvice(
            destination=destination,
            advice=f"Error planning trip: {str(e)}"
        )

async def main():
    destination = "Paris, France"
    print(f"\nPlanning a trip to {destination}...")
    
    try:
        # Get travel advice
        result = await travel_planner(destination)
        
        print("\nTravel Advice:")
        print("-" * 50)
        print(result.advice)
        print("-" * 50)
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())