import json
import logging
from typing import Dict, Any

def generate_mcq_with_groq(chunk: str, question: str) -> Dict[str, Any]:
    """Generate MCQ using Groq API with improved error handling and JSON sanitization."""
    if not groq_client:
        raise ValueError("Groq client not available")
    
    try:
        messages = get_mcq_prompt(chunk, question)
        
        completion = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=messages,
            temperature=Config.DEFAULT_TEMPERATURE,
            max_tokens=Config.DEFAULT_MAX_TOKENS,
            top_p=0.9,
            stream=False
        )
        
        response_content = completion.choices[0].message.content.strip()
        
        if response_content.startswith('```json'):
            response_content = response_content.replace('```json', '').replace('```', '').strip()
        elif response_content.startswith('```'):
            response_content = response_content.replace('```', '').strip()
        
        
        json_start = response_content.find('{')
        if json_start != -1:
            brace_count = 0
            json_end = -1
            for i in range(json_start, len(response_content)):
                if response_content[i] == '{':
                    brace_count += 1
                elif response_content[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i + 1
                        break
            
            if json_end != -1:
                response_content = response_content[json_start:json_end]
        
        try:
            parsed_response = json.loads(response_content)
            
            if isinstance(parsed_response, dict):
                validated_response = {
                    'question': str(parsed_response.get('question', question))[:300],
                    'options': [],
                    'answer': int(parsed_response.get('answer', 1)),
                    'explanation': str(parsed_response.get('explanation', 'No explanation provided.'))[:500]
                }
                
                raw_options = parsed_response.get('options', [])
                if isinstance(raw_options, list) and len(raw_options) >= 4:
                    options = [str(opt)[:100].strip() for opt in raw_options[:4]]
                    generic_patterns = ['option a', 'option b', 'option c', 'option d', 'choice a', 'choice b']
                    has_generic = any(any(pattern in opt.lower() for pattern in generic_patterns) for opt in options)
                    
                    if has_generic:
                        logger.warning("Detected generic options, regenerating...")
                        raise ValueError("Generic options detected")
                    
                    validated_response['options'] = options
                else:
                    logger.warning("Invalid options structure, regenerating...")
                    raise ValueError("Invalid options structure")
                
                if not (1 <= validated_response['answer'] <= 4):
                    validated_response['answer'] = 1
                
                explanation = validated_response['explanation']
                if '{' in explanation and '"' in explanation:
                    try:
                        if '"options"' in explanation:
                            import re
                            options_match = re.search(r'"options":\s*\[(.*?)\]', explanation, re.DOTALL)
                            if options_match:
                                options_text = options_match.group(1)
                                options = []
                                for match in re.finditer(r'"([^"]*)"', options_text):
                                    option_text = match.group(1)
                                    if len(option_text) > 5 and not option_text.lower().startswith('option'):
                                        options.append(option_text[:100])
                                
                                if len(options) >= 4:
                                    validated_response['options'] = options[:4]
                                    explanation = f"Based on the content provided, the correct answer relates to the main topic discussed."
                        
                            explanation = "Based on the content analysis, this answer is most appropriate."
                    except:
                        explanation = "Answer determined from content analysis."
                
                explanation = explanation.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
                if explanation.startswith('{') or explanation.endswith('}'):
                    explanation = "Answer based on the provided content."
                
                validated_response['explanation'] = explanation[:500]
                
                return validated_response
            else:
                raise ValueError("Response is not a dictionary")
                
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse Groq response as JSON or detected generic options: {e}")
            logger.warning(f"Raw response: {response_content[:200]}...")
            
            if "Generic options detected" in str(e) or "Invalid options structure" in str(e):
                try:
                    retry_messages = [
                        {
                            "role": "user", 
                            "content": f"""The content discusses: {chunk[:200]}...

Create a multiple-choice question about: {question}

You MUST provide 4 REAL, SPECIFIC answer choices - NO generic options like "Option A".

Example of what NOT to do: ["Option A", "Option B", "Option C", "Option D"]
Example of what TO do: ["Renewable energy sources", "Fossil fuel dependency", "Nuclear power plants", "Hydroelectric systems"]

Return only JSON:
{{
    "question": "specific question here?",
    "options": ["real answer 1", "real answer 2", "real answer 3", "real answer 4"],
    "answer": 2,
    "explanation": "why answer 2 is correct"
}}"""
                        }
                    ]
                    
                    retry_completion = groq_client.chat.completions.create(
                        model="llama3-70b-8192",
                        messages=retry_messages,
                        temperature=0.7,
                        max_tokens=500,
                        stream=False
                    )
                    
                    retry_content = retry_completion.choices[0].message.content.strip()
                    if retry_content.startswith('```json'):
                        retry_content = retry_content.replace('```json', '').replace('```', '').strip()
                    
                    retry_parsed = json.loads(retry_content)
                    retry_options = retry_parsed.get('options', [])
                    
                    retry_options = retry_parsed.get('options', [])
                    retry_explanation = str(retry_parsed.get('explanation', ''))
                    
                    if any(any(pattern in str(opt).lower() for pattern in generic_patterns) for opt in retry_options):
                        if '"options"' in retry_explanation and '[' in retry_explanation:
                            import re
                            options_match = re.search(r'"options":\s*\[(.*?)\]', retry_explanation, re.DOTALL)
                            if options_match:
                                options_text = options_match.group(1)
                                extracted_options = []
                                for match in re.finditer(r'"([^"]*)"', options_text):
                                    option_text = match.group(1)
                                    if len(option_text) > 5 and not any(p in option_text.lower() for p in generic_patterns):
                                        extracted_options.append(option_text[:100])
                                
                                if len(extracted_options) >= 4:
                                    retry_options = extracted_options[:4]
                    
                    has_generic = any(any(pattern in str(opt).lower() for pattern in generic_patterns) for opt in retry_options)
                    
                    if not has_generic and len(retry_options) >= 4:
                        validated_response = {
                            'question': str(retry_parsed.get('question', question))[:300],
                            'options': [str(opt)[:100].strip() for opt in retry_options[:4]],
                            'answer': int(retry_parsed.get('answer', 1)),
                            'explanation': "Answer determined based on content analysis."
                        }
                        
                        if not (1 <= validated_response['answer'] <= 4):
                            validated_response['answer'] = 1
                            
                        return validated_response
                        
                except Exception as retry_error:
                    logger.warning(f"Retry attempt failed: {retry_error}")
            
            context_keywords = chunk[:300].lower()
            fallback_options = ["Unable to generate specific options", "Content processing incomplete", "Answer not determinable", "Information insufficient"]
            
            if any(word in context_keywords for word in ['technology', 'digital', 'computer']):
                fallback_options = ["Advanced technology solutions", "Traditional manual methods", "Outdated legacy systems", "Basic communication tools"]
            elif any(word in context_keywords for word in ['medicine', 'health', 'treatment']):
                fallback_options = ["Personalized medical treatments", "Generic healthcare approaches", "Traditional remedies only", "Experimental procedures"]
            elif any(word in context_keywords for word in ['energy', 'power', 'electric']):
                fallback_options = ["Renewable energy sources", "Fossil fuel dependency", "Nuclear power generation", "Manual energy production"]
            
            return {
                "question": question,
                "options": fallback_options,
                "answer": 1,
                "explanation": "Unable to generate proper MCQ options, using context-based fallback."
            }
    
    except Exception as e:
        logger.error(f"Error generating MCQ with Groq: {e}")
        return {
            "question": question,
            "options": ["Unable to generate options due to error"],
            "answer": 1,
            "explanation": f"Error occurred: {str(e)[:100]}",
            "error": str(e)
        }


def process_pdf_and_generate_qa_stream(pdf_path: str, num_questions: int):
    """Main processing function with streaming response and improved JSON handling."""
    try:
        text = extract_text_from_pdf(pdf_path)
        chunks = split_text_into_chunks(text)
        
        max_chunks = min(len(chunks), num_questions)
        
        for i in range(max_chunks):
            try:
                chunk = chunks[i]
                
                question = generate_questions_t5(chunk, model, tokenizer)
                
                mcq_data = generate_mcq_with_groq(chunk, question)
                
                mcq_data.update({
                    "chunk_id": i + 1,
                    "total_chunks": max_chunks,
                    "source_chunk": chunk[:100] + "..." if len(chunk) > 100 else chunk
                })
                
                try:
                    json.dumps(mcq_data) 
                    yield mcq_data
                except (TypeError, ValueError) as json_error:
                    logger.error(f"JSON serialization error for chunk {i+1}: {json_error}")
                    yield {
                        "question": f"Question {i+1}: Content processing completed",
                        "options": ["Option A", "Option B", "Option C", "Option D"],
                        "answer": 1,
                        "explanation": "Content was processed but formatting encountered issues.",
                        "chunk_id": i + 1,
                        "total_chunks": max_chunks,
                        "error": "JSON formatting issue resolved"
                    }
                
            except Exception as e:
                logger.error(f"Error processing chunk {i+1}: {e}")
                yield {
                    "question": f"Error processing question {i+1}",
                    "options": ["Error occurred during processing"],
                    "answer": 1,
                    "explanation": f"An error occurred: {str(e)[:100]}",
                    "chunk_id": i + 1,
                    "total_chunks": max_chunks,
                    "error": str(e)
                }
    
    except Exception as e:
        logger.error(f"Error in PDF processing: {e}")
        yield {
            "error": str(e),
            "question": "Processing failed",
            "options": ["Error occurred"],
            "answer": 1,
            "explanation": f"PDF processing failed: {str(e)[:100]}"
        }

def get_mcq_prompt(chunk, question):
    """Enhanced prompt with stronger enforcement of proper MCQ generation."""
    return [
        {
            "role": "system",
            "content": """You are an expert MCQ creator. You MUST create 4 specific, meaningful answer options - never use generic placeholders like "Option A" or "Option B". Always return valid JSON only."""
        },
        {
            "role": "user",
            "content": f"""Based on this content, create a multiple-choice question with 4 SPECIFIC answer options.

CONTENT:
{chunk[:1000]}

ORIGINAL QUESTION: {question}

CRITICAL REQUIREMENTS:
1. Create 4 SPECIFIC, MEANINGFUL answer options (never "Option A", "Option B", etc.)
2. Each option must be a real answer attempt, not a placeholder
3. Make options 3-12 words each
4. Only ONE option is correct based on the content
5. Make wrong options believable but clearly incorrect
6. Place correct answer randomly in positions 1-4

EXAMPLE FORMAT (use real content, not these examples):
{{
    "question": "What technology revolutionized manufacturing in the 1800s?",
    "options": [
        "Steam engines and mechanization",
        "Computer programming",
        "Solar panel technology", 
        "Internet connectivity"
    ],
    "answer": 1,
    "explanation": "Steam engines and mechanization were the key technologies of the Industrial Revolution."
}}

NOW CREATE YOUR MCQ - Return ONLY the JSON object with NO extra text:"""
        }
    ]