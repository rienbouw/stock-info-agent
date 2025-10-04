import json
from typing import Optional, Dict, Any, List
from openai import OpenAI
import yfinance as yf
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

class StockInfoAgent:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.conversation_history = []
        
    def get_stock_price(self, ticker_symbol: str) -> Optional[str]:
        """Fetches the current stock price for the given ticker_symbol.

        Returns prices converted to EUR where possible. The method retrieves the
        stock's reported currency and price, then looks up the forex rate using
        yfinance (ticker like "EURUSD=X" for USD->EUR). If FX lookup fails the
        original price with its currency is returned.
        """
        try:
            stock = yf.Ticker(ticker_symbol.upper())
            info = stock.info
            # Try to read the current price and the reported currency
            current_price = info.get('currentPrice') or info.get('regularMarketPrice')
            currency = info.get('currency', 'USD')

            if current_price is None:
                return None

            # If the price is already in EUR, just format and return
            if currency == 'EUR':
                return f"{current_price:.2f} EUR"

            # Otherwise, convert to EUR using forex ticker. Many tickers are quoted in their local
            # currency (e.g., USD). We fetch the appropriate FX rate with yfinance. We expect
            # forex tickers in the form: {FROM}{TO}=X, for example USD->EUR is EURUSD=X which gives
            # USD per 1 EUR. To convert a price quoted in USD to EUR: price_in_eur = price_in_usd / (USD per EUR)
            try:
                # Build forex ticker: if currency is USD, use EURUSD=X; for other currencies use EUR{currency}=X
                forex_symbol = f"EUR{currency}=X"
                fx = yf.Ticker(forex_symbol)
                fx_info = fx.info
                fx_rate = fx_info.get('regularMarketPrice') or fx_info.get('previousClose') or fx_info.get('currentPrice')

                if fx_rate:
                    # fx_rate is amount of 'currency' per 1 EUR. To convert: price_in_eur = price_in_currency / fx_rate
                    price_eur = float(current_price) / float(fx_rate)
                    return f"{price_eur:.2f} EUR"
            except Exception:
                # Fall back to returning the original price with its currency if FX failed
                return f"{current_price:.2f} {currency}"

            # If FX rate not found, return original with currency
            return f"{current_price:.2f} {currency}"
        except Exception as e:
            print(f"Fout bij ophalen aandelenkoers: {e}")
            return None
    
    def get_company_ceo(self, ticker_symbol: str) -> Optional[str]:
        """Fetches the name of the CEO for the company associated with the ticker_symbol."""
        try:
            stock = yf.Ticker(ticker_symbol.upper())
            info = stock.info
            
            # Look for CEO in various possible fields
            ceo = None
            for field in ['companyOfficers', 'officers']:
                if field in info:
                    officers = info[field]
                    if isinstance(officers, list):
                        for officer in officers:
                            if isinstance(officer, dict):
                                title = officer.get('title', '').lower()
                                if 'ceo' in title or 'chief executive' in title:
                                    ceo = officer.get('name')
                                    break
            
            # Fallback to general company info
            if not ceo and 'longBusinessSummary' in info:
                ceo = None  
                
            return ceo
        except Exception as e:
            print(f"Fout bij ophalen CEO-informatie: {e}")
            return None
    
    def find_ticker_symbol(self, company_name: str) -> Optional[str]:
        """Tries to identify the stock ticker symbol for a given company_name."""
        try:
            # Use yfinance Lookup to search for the company
            lookup = yf.Lookup(company_name)
            
            stock_results = lookup.get_stock(count=5)
            
            if not stock_results.empty:
                return stock_results.index[0]
            
            # If no stocks found, try all instruments
            all_results = lookup.get_all(count=5)
            
            if not all_results.empty:
                return all_results.index[0]
                
        except Exception as e:
            print(f"Error searching for ticker: {e}")
        
        return None
    
    def ask_user_for_clarification(self, question_to_user: str) -> str:
        """Poses the question_to_user to the actual user and returns their typed response."""
        print(f"\nAgent heeft verduidelijking nodig: {question_to_user}")
        response = input("Jouw antwoord: ")
        return response
    
    def create_tool_definitions(self) -> List[Dict[str, Any]]:
        """Creates OpenAI function calling definitions for the tools."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_stock_price",
                    "description": "Haalt de huidige aandelenkoers op voor het gegeven ticker-symbool",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "ticker_symbol": {
                                "type": "string",
                                "description": "Het aandelen-ticker symbool (bijv. 'AAPL', 'MSFT')"
                            }
                        },
                        "required": ["ticker_symbol"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_company_ceo",
                    "description": "Haalt de naam van de CEO op voor het bedrijf dat bij het ticker-symbool hoort",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "ticker_symbol": {
                                "type": "string",
                                "description": "Het aandelen-ticker symbool"
                            }
                        },
                        "required": ["ticker_symbol"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "find_ticker_symbol",
                    "description": "Probeert het aandelen-ticker symbool te vinden voor een gegeven bedrijfsnaam",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "company_name": {
                                "type": "string",
                                "description": "De naam van het bedrijf"
                            }
                        },
                        "required": ["company_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "ask_user_for_clarification",
                    "description": "Stelt een vraag aan de gebruiker en retourneert hun antwoord",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question_to_user": {
                                "type": "string",
                                "description": "De vraag die aan de gebruiker gesteld moet worden"
                            }
                        },
                        "required": ["question_to_user"]
                    }
                }
            }
        ]
    
    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Executes the specified tool with given arguments."""
        func = getattr(self, tool_name, None)
        if not callable(func):
            return None

        if arguments:
            first_value = next(iter(arguments.values()))
            return func(first_value)
        else:
            return func()
    
    def process_user_query(self, user_query: str) -> str:
        """Processes a user query using the OpenAI API with function calling."""
        # Add user message to conversation history
        self.conversation_history.append({"role": "user", "content": user_query})
        
        system_prompt = """Je bent een behulpzame assistent voor aandeleninformatie. Je hebt toegang tot hulpmiddelen die het volgende kunnen:
                        1. Huidige aandelenkoersen opvragen
                        2. De CEO van een bedrijf vinden
                        3. Ticker-symbolen vinden voor bedrijfsnamen
                        4. De gebruiker om verduidelijking vragen wanneer dat nodig is

                        Gebruik deze hulpmiddelen om gebruikersvragen over aandelen en bedrijven te beantwoorden. Als informatie onduidelijk is, vraag dan om verduidelijking."""
        
        while True:
            messages = [
                {"role": "system", "content": system_prompt},
                *self.conversation_history
            ]
            
            # Call OpenAI API with function calling
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=messages,
                tools=self.create_tool_definitions(),
                tool_choice="auto"
            )
            
            response_message = response.choices[0].message
            
            # If no tool calls, we're done
            if not response_message.tool_calls:
                self.conversation_history.append({"role": "assistant", "content": response_message.content})
                return response_message.content
            
            # Execute the first tool call
            tool_call = response_message.tool_calls[0]
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            print(f"\nUitvoeren tool: {function_name} met argumenten: {function_args}")
            
            # Execute the tool
            result = self.execute_tool(function_name, function_args)
            
            # Add the assistant's message with tool calls to history
            self.conversation_history.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "arguments": json.dumps(function_args)
                    }
                }]
            })
            
            # Add tool result to history
            self.conversation_history.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": str(result) if result is not None else "No result found"
            })
    
    def chat(self):
        """Interactive chat loop."""
        print("Aandelen Informatie Agent")
        print("Vraag me naar aandelenkoersen, CEO's van bedrijven, of andere aandelen-gerelateerde vragen!")
        print("Typ 'quit' om af te sluiten.\n")
        
        while True:
            user_input = input("Jij: ")
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Tot ziens!")
                break
            
            try:
                response = self.process_user_query(user_input)
                print(f"\nAgent: {response}\n")
            except Exception as e:
                print(f"\nError: {e}\n")

if __name__ == "__main__":
    agent = StockInfoAgent()
    agent.chat()