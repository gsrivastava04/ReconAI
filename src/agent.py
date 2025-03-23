import os
import pandas as pd
import numpy as np
import logging
import json
import openai
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum

from config import OPENAI_API_KEY, ACTION_TYPES, ANOMALY_CATEGORIES

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ActionType(Enum):
    """Enum for types of actions the agent can take"""
    CREATE_JIRA = "Create JIRA Ticket"
    SEND_EMAIL = "Send Email"
    UPDATE_SOURCE = "Update Source System"
    DOCUMENT_EXCEPTION = "Document Exception"
    REPROCESS = "Reprocess Reconciliation"
    ESCALATE = "Escalate to Team Lead"
    NO_ACTION = "No Action Required"

class ReconciliationAgent:
    """
    AI agent that assists with reconciliation break resolution
    by analyzing anomalies and suggesting or taking actions
    """
    
    def __init__(self, model: str = "gpt-3.5-turbo", human_in_loop: bool = True):
        """
        Initialize the reconciliation agent
        
        Args:
            model (str): The LLM model to use
            human_in_loop (bool): Whether to require human approval for actions
        """
        self.model = model
        self.human_in_loop = human_in_loop
        self.feedback_history = []
        self.action_history = []
        
        # API setup
        if OPENAI_API_KEY:
            openai.api_key = OPENAI_API_KEY
        else:
            logger.warning("OpenAI API key not provided. Agent will use rule-based approach only.")
        
    def analyze_break(self, record: Dict[str, Any], anomaly_category: str = None) -> Dict[str, Any]:
        """
        Analyze a reconciliation break and provide insights and suggested actions
        
        Args:
            record (Dict[str, Any]): The break record
            anomaly_category (str, optional): The category of anomaly if known
            
        Returns:
            Dict[str, Any]: Analysis results including summary and suggested actions
        """
        if not OPENAI_API_KEY:
            return self._rule_based_analysis(record, anomaly_category)
        
        try:
            # Format the record for better readability
            formatted_record = json.dumps(record, indent=2)
            
            # Create a prompt for the LLM
            prompt = f"""
            You are a financial reconciliation expert AI assistant. You're analyzing a reconciliation break.
            
            BREAK RECORD:
            {formatted_record}
            
            {"ANOMALY CATEGORY: " + anomaly_category if anomaly_category else ""}
            
            Please analyze this break and provide:
            1. A brief summary of what appears to be causing the break (2-3 sentences)
            2. The most likely root cause
            3. Recommended action to resolve the break (select one): {', '.join(ACTION_TYPES)}
            4. Detailed justification for your recommendation
            
            Format your response as a JSON object with the following keys:
            {{
              "summary": "...",
              "root_cause": "...",
              "recommended_action": "...",
              "justification": "..."
            }}
            """
            
            # Call the LLM API
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a financial reconciliation AI assistant that helps analyze and resolve breaks between financial systems."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1000
            )
            
            # Parse the response
            response_text = response.choices[0].message.content.strip()
            
            # Extract JSON from response
            try:
                # Find JSON in the response
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    analysis = json.loads(json_str)
                else:
                    # Fallback to rule-based if JSON not found
                    logger.warning("No valid JSON found in LLM response")
                    return self._rule_based_analysis(record, anomaly_category)
                
                # Validate the expected fields
                required_fields = ["summary", "root_cause", "recommended_action", "justification"]
                for field in required_fields:
                    if field not in analysis:
                        analysis[field] = "Not provided"
                
                # Make sure the action is one of the valid types
                if analysis["recommended_action"] not in ACTION_TYPES:
                    analysis["recommended_action"] = "Document Exception"
                
                return analysis
            
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON from LLM response")
                return self._rule_based_analysis(record, anomaly_category)
                
        except Exception as e:
            logger.error(f"Error during break analysis with LLM: {str(e)}")
            return self._rule_based_analysis(record, anomaly_category)
    
    def _rule_based_analysis(self, record: Dict[str, Any], anomaly_category: str = None) -> Dict[str, Any]:
        """
        Fallback rule-based analysis when LLM is not available
        
        Args:
            record (Dict[str, Any]): The break record
            anomaly_category (str, optional): The category of anomaly if known
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        analysis = {
            "summary": "Break detected in reconciliation data.",
            "root_cause": "Unknown - rule-based analysis applied",
            "recommended_action": "Document Exception",
            "justification": "Automated rule-based analysis without LLM."
        }
        
        # Adjust based on anomaly category if provided
        if anomaly_category:
            if anomaly_category == "Missing Data":
                analysis["summary"] = "Break caused by missing data in one of the systems."
                analysis["root_cause"] = "Missing Data"
                analysis["recommended_action"] = "Update Source System"
                analysis["justification"] = "Missing data needs to be populated in the source system."
                
            elif anomaly_category == "Timing Difference":
                analysis["summary"] = "Break caused by timing difference between systems."
                analysis["root_cause"] = "Timing Difference"
                analysis["recommended_action"] = "Document Exception"
                analysis["justification"] = "This is a timing issue that will likely resolve in the next reconciliation cycle."
                
            elif anomaly_category == "Calculation Error":
                analysis["summary"] = "Break caused by calculation difference between systems."
                analysis["root_cause"] = "Calculation Error"
                analysis["recommended_action"] = "Create JIRA Ticket"
                analysis["justification"] = "Investigation needed to identify and fix the calculation discrepancy."
                
        # Check for large differences that might need escalation
        if 'Balance Difference' in record and abs(float(record.get('Balance Difference', 0))) > 10000:
            analysis["recommended_action"] = "Escalate to Team Lead"
            analysis["justification"] = "Large balance difference requires supervisor review."
            
        # Check for weekend transactions
        date_fields = [k for k in record.keys() if 'date' in k.lower()]
        for field in date_fields:
            try:
                date_val = pd.to_datetime(record[field])
                if date_val.dayofweek >= 5:  # Weekend
                    analysis["summary"] = "Weekend transaction detected, likely timing issue."
                    analysis["root_cause"] = "Weekend Processing"
                    analysis["recommended_action"] = "Document Exception"
                    analysis["justification"] = "Weekend transactions often cause temporary reconciliation breaks."
                    break
            except:
                pass
                
        return analysis
    
    def take_action(self, record_id: str, action_type: str, details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute or simulate the recommended action
        
        Args:
            record_id (str): Identifier for the record
            action_type (str): Type of action to take
            details (Dict[str, Any]): Details about the action
            
        Returns:
            Dict[str, Any]: Result of the action
        """
        # Check if human approval is required
        if self.human_in_loop:
            # This would typically prompt the user in the UI
            # For now, we'll simulate approval
            logger.info(f"[HUMAN-IN-LOOP] Action {action_type} for record {record_id} requires approval")
            
        result = {
            "record_id": record_id,
            "action_type": action_type,
            "status": "completed",
            "timestamp": pd.Timestamp.now().isoformat(),
            "details": {}
        }
        
        # Simulate different action types
        if action_type == ActionType.CREATE_JIRA.value:
            # Simulate JIRA ticket creation
            result["details"] = {
                "ticket_id": f"RECON-{int(time.time())}",
                "summary": details.get("summary", "Reconciliation break detected"),
                "description": details.get("justification", "Investigation required"),
                "status": "Open"
            }
            
        elif action_type == ActionType.SEND_EMAIL.value:
            # Simulate email sending
            recipients = details.get("recipients", ["reconciliation_team@example.com"])
            result["details"] = {
                "to": recipients,
                "subject": f"Reconciliation Break Alert: {record_id}",
                "body": details.get("justification", "Reconciliation break detected"),
                "sent": True
            }
            
        elif action_type == ActionType.UPDATE_SOURCE.value:
            # Simulate source system update
            result["details"] = {
                "system": details.get("system", "Unknown"),
                "field": details.get("field", "Unknown"),
                "old_value": details.get("old_value", "Unknown"),
                "new_value": details.get("new_value", "Unknown"),
                "status": "Updated"
            }
            
        elif action_type == ActionType.DOCUMENT_EXCEPTION.value:
            # Simulate documenting the exception
            result["details"] = {
                "exception_id": f"EXC-{int(time.time())}",
                "reason": details.get("justification", "Exception documented"),
                "status": "Documented"
            }
            
        elif action_type == ActionType.REPROCESS.value:
            # Simulate reprocessing request
            result["details"] = {
                "reprocess_id": f"REP-{int(time.time())}",
                "status": "Queued",
                "scheduled_time": (pd.Timestamp.now() + pd.Timedelta(minutes=30)).isoformat()
            }
            
        elif action_type == ActionType.ESCALATE.value:
            # Simulate escalation
            result["details"] = {
                "escalation_id": f"ESC-{int(time.time())}",
                "escalated_to": details.get("escalate_to", "Team Lead"),
                "priority": "High",
                "status": "Pending Review"
            }
            
        else:  # No action
            result["details"] = {
                "note": "No action required",
                "reason": details.get("justification", "No intervention needed")
            }
        
        # Log the action
        logger.info(f"Action taken: {action_type} for record {record_id}")
        
        # Store in history
        self.action_history.append(result)
        
        return result
    
    def learn_from_feedback(self, record: Dict[str, Any], analysis: Dict[str, Any], 
                           actual_resolution: Dict[str, Any], feedback: str) -> None:
        """
        Store feedback from reconcilers to improve future recommendations
        
        Args:
            record (Dict[str, Any]): The original record
            analysis (Dict[str, Any]): The agent's analysis
            actual_resolution (Dict[str, Any]): What actually resolved the break
            feedback (str): Feedback from the user
        """
        feedback_entry = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "record": record,
            "agent_analysis": analysis,
            "actual_resolution": actual_resolution,
            "feedback": feedback
        }
        
        # Store feedback for learning
        self.feedback_history.append(feedback_entry)
        
        # For a production system, you'd periodically use this feedback to:
        # 1. Fine-tune the LLM with these examples
        # 2. Adjust rule-based logic based on patterns
        # 3. Update prompt templates to improve performance
        
        logger.info(f"Feedback recorded: {feedback}")
    
    def get_resolution_summary(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a summary of break resolutions for a batch of records
        
        Args:
            records (List[Dict[str, Any]]): List of break records
            
        Returns:
            Dict[str, Any]: Summary statistics and insights
        """
        if not records:
            return {"error": "No records provided"}
        
        if not OPENAI_API_KEY:
            return self._generate_basic_summary(records)
        
        try:
            # Prepare a condensed version of records for the prompt
            record_summaries = []
            for i, record in enumerate(records[:10]):  # Limit to first 10 for API context window
                # Extract key fields for each record
                summary = {
                    "id": record.get("id", f"Record_{i}"),
                    "type": record.get("anomaly_category", "Unknown"),
                    "balance_difference": record.get("Balance Difference", "N/A"),
                    "action_taken": record.get("action_taken", "No action recorded")
                }
                record_summaries.append(summary)
            
            # Create the prompt
            prompt = f"""
            You are a financial reconciliation expert. Analyze these {len(records)} reconciliation breaks and provide:
            1. A concise executive summary (3-5 sentences)
            2. Key statistics about the breaks
            3. Patterns or trends you observe
            4. Recommendations for process improvement
            
            Here's a sample of the break data:
            {json.dumps(record_summaries, indent=2)}
            
            Format your response as a JSON object with the following keys:
            {{
              "executive_summary": "...",
              "key_statistics": {{...}},
              "patterns_observed": [...],
              "recommendations": [...]
            }}
            """
            
            # Call the LLM API
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a financial reconciliation AI assistant that helps analyze patterns in reconciliation breaks."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1500
            )
            
            # Parse the response
            response_text = response.choices[0].message.content.strip()
            
            try:
                # Find JSON in the response
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    summary = json.loads(json_str)
                    return summary
                else:
                    logger.warning("No valid JSON found in LLM summary response")
                    return self._generate_basic_summary(records)
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON from LLM summary response")
                return self._generate_basic_summary(records)
                
        except Exception as e:
            logger.error(f"Error generating resolution summary with LLM: {str(e)}")
            return self._generate_basic_summary(records)
    
    def _generate_basic_summary(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a basic statistical summary when LLM is not available
        
        Args:
            records (List[Dict[str, Any]]): List of break records
            
        Returns:
            Dict[str, Any]: Basic summary statistics
        """
        # Count records by anomaly category
        category_counts = {}
        for record in records:
            category = record.get("anomaly_category", "Unknown")
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Count actions taken
        action_counts = {}
        for record in records:
            action = record.get("action_taken", "No Action")
            action_counts[action] = action_counts.get(action, 0) + 1
        
        # Calculate average, min, max balance differences
        balance_diffs = []
        for record in records:
            if "Balance Difference" in record:
                try:
                    diff = float(record["Balance Difference"])
                    balance_diffs.append(diff)
                except (ValueError, TypeError):
                    pass
        
        balance_stats = {}
        if balance_diffs:
            balance_stats = {
                "average": np.mean(balance_diffs),
                "min": np.min(balance_diffs),
                "max": np.max(balance_diffs),
                "std_dev": np.std(balance_diffs)
            }
        
        return {
            "executive_summary": f"Analyzed {len(records)} reconciliation breaks. Most common category: {max(category_counts.items(), key=lambda x: x[1])[0] if category_counts else 'Unknown'}.",
            "key_statistics": {
                "total_breaks": len(records),
                "by_category": category_counts,
                "by_action": action_counts,
                "balance_differences": balance_stats
            },
            "patterns_observed": [
                "Basic statistical analysis performed (LLM-based pattern recognition not available)"
            ],
            "recommendations": [
                "Review the most common break categories to identify process improvements",
                "Consider addressing systemic issues that may be causing repeated breaks"
            ]
        }