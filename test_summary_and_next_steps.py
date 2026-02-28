"""
Comprehensive unit tests for 9_summary_and_next_steps.ipynb

This test file covers:
- Individual node functions (topics_review_node, skills_assessment_node, next_steps_planner_node, reflection_node)
- State management and TypedDict structure
- Graph construction and compilation
- Edge cases and error scenarios
"""

import unittest
from typing import TypedDict, List
from unittest.mock import patch, MagicMock
import sys

# Import the functions and classes from the notebook
# Since it's in a notebook, we'll define them here for testing
from langgraph.graph import StateGraph, END


# Define state for our summary graph (from notebook)
class SummaryState(TypedDict):
    topics_covered: List[str]
    skills_acquired: List[str]
    next_steps: List[str]
    final_thoughts: str


# Define nodes for our summary graph (from notebook)
def topics_review_node(state):
    """Review topics covered in the course"""
    topics = [
        "LangGraph fundamentals and architecture",
        "State management and TypedDict",
        "Node creation and graph construction",
        "Conditional logic and branching",
        "Parallel processing and concurrency",
        "Advanced patterns (recursion, decision trees)",
        "Real-world application patterns",
        "Testing and debugging strategies",
        "Deployment and production considerations",
        "Security best practices"
    ]

    return {
        "topics_covered": topics
    }


def skills_assessment_node(state):
    """Assess skills acquired during the course"""
    skills = [
        "Designing stateful graph applications",
        "Implementing complex business logic with LangGraph",
        "Building conversational AI systems",
        "Creating multi-modal processing pipelines",
        "Implementing planning and execution systems",
        "Testing and debugging graph applications",
        "Productionizing LangGraph applications",
        "Applying security best practices to graphs"
    ]

    return {
        "skills_acquired": skills
    }


def next_steps_planner_node(state):
    """Plan next steps for continued learning"""
    next_steps = [
        "Explore advanced LangGraph features like streaming and async processing",
        "Integrate LangGraph with external APIs and databases",
        "Build custom LangGraph components and tools",
        "Experiment with memory mechanisms in LangGraph",
        "Learn about LangGraph optimization techniques",
        "Contribute to LangGraph open-source community",
        "Apply LangGraph to domain-specific problems",
        "Study advanced patterns like reflection and tool-use"
    ]

    return {
        "next_steps": next_steps
    }


def reflection_node(state):
    """Reflect on the learning journey"""
    reflection = "LangGraph represents a paradigm shift in building AI applications. " \
                 "By structuring applications as graphs, we can handle complex, multi-step " \
                 "reasoning while maintaining clarity and control. The framework excels at " \
                 "building stateful applications that require memory of past interactions, " \
                 "making it ideal for agents, conversational systems, and complex workflows. " \
                 "The combination of LangChain's ecosystem with graph-based architectures " \
                 "opens new possibilities for creating sophisticated AI applications that " \
                 "can reason, plan, and adapt to dynamic situations."

    return {
        "final_thoughts": reflection
    }


class TestTopicsReviewNode(unittest.TestCase):
    """Test suite for topics_review_node function"""

    def test_topics_review_returns_list(self):
        """Test that topics_review_node returns a list of topics"""
        result = topics_review_node({})
        self.assertIn("topics_covered", result)
        self.assertIsInstance(result["topics_covered"], list)

    def test_topics_review_has_expected_count(self):
        """Test that topics_review_node returns expected number of topics"""
        result = topics_review_node({})
        self.assertEqual(len(result["topics_covered"]), 10)

    def test_topics_review_contains_langgraph_fundamentals(self):
        """Test that topics include LangGraph fundamentals"""
        result = topics_review_node({})
        topics = result["topics_covered"]
        self.assertIn("LangGraph fundamentals and architecture", topics)

    def test_topics_review_contains_state_management(self):
        """Test that topics include state management"""
        result = topics_review_node({})
        topics = result["topics_covered"]
        self.assertIn("State management and TypedDict", topics)

    def test_topics_review_all_strings(self):
        """Test that all topics are strings"""
        result = topics_review_node({})
        topics = result["topics_covered"]
        self.assertTrue(all(isinstance(topic, str) for topic in topics))

    def test_topics_review_no_empty_strings(self):
        """Test that no topics are empty strings"""
        result = topics_review_node({})
        topics = result["topics_covered"]
        self.assertTrue(all(len(topic) > 0 for topic in topics))

    def test_topics_review_with_existing_state(self):
        """Test that node works with pre-existing state"""
        state = {
            "topics_covered": ["existing topic"],
            "skills_acquired": [],
            "next_steps": [],
            "final_thoughts": ""
        }
        result = topics_review_node(state)
        self.assertEqual(len(result["topics_covered"]), 10)

    def test_topics_review_idempotent(self):
        """Test that calling the function multiple times produces same result"""
        result1 = topics_review_node({})
        result2 = topics_review_node({})
        self.assertEqual(result1["topics_covered"], result2["topics_covered"])


class TestSkillsAssessmentNode(unittest.TestCase):
    """Test suite for skills_assessment_node function"""

    def test_skills_assessment_returns_list(self):
        """Test that skills_assessment_node returns a list of skills"""
        result = skills_assessment_node({})
        self.assertIn("skills_acquired", result)
        self.assertIsInstance(result["skills_acquired"], list)

    def test_skills_assessment_has_expected_count(self):
        """Test that skills_assessment_node returns expected number of skills"""
        result = skills_assessment_node({})
        self.assertEqual(len(result["skills_acquired"]), 8)

    def test_skills_assessment_contains_graph_design(self):
        """Test that skills include designing graph applications"""
        result = skills_assessment_node({})
        skills = result["skills_acquired"]
        self.assertIn("Designing stateful graph applications", skills)

    def test_skills_assessment_contains_conversational_ai(self):
        """Test that skills include building conversational AI"""
        result = skills_assessment_node({})
        skills = result["skills_acquired"]
        self.assertIn("Building conversational AI systems", skills)

    def test_skills_assessment_all_strings(self):
        """Test that all skills are strings"""
        result = skills_assessment_node({})
        skills = result["skills_acquired"]
        self.assertTrue(all(isinstance(skill, str) for skill in skills))

    def test_skills_assessment_no_empty_strings(self):
        """Test that no skills are empty strings"""
        result = skills_assessment_node({})
        skills = result["skills_acquired"]
        self.assertTrue(all(len(skill) > 0 for skill in skills))

    def test_skills_assessment_with_existing_state(self):
        """Test that node works with pre-existing state"""
        state = {
            "topics_covered": [],
            "skills_acquired": ["existing skill"],
            "next_steps": [],
            "final_thoughts": ""
        }
        result = skills_assessment_node(state)
        self.assertEqual(len(result["skills_acquired"]), 8)

    def test_skills_assessment_idempotent(self):
        """Test that calling the function multiple times produces same result"""
        result1 = skills_assessment_node({})
        result2 = skills_assessment_node({})
        self.assertEqual(result1["skills_acquired"], result2["skills_acquired"])


class TestNextStepsPlannerNode(unittest.TestCase):
    """Test suite for next_steps_planner_node function"""

    def test_next_steps_returns_list(self):
        """Test that next_steps_planner_node returns a list of next steps"""
        result = next_steps_planner_node({})
        self.assertIn("next_steps", result)
        self.assertIsInstance(result["next_steps"], list)

    def test_next_steps_has_expected_count(self):
        """Test that next_steps_planner_node returns expected number of steps"""
        result = next_steps_planner_node({})
        self.assertEqual(len(result["next_steps"]), 8)

    def test_next_steps_contains_streaming(self):
        """Test that next steps include streaming and async processing"""
        result = next_steps_planner_node({})
        steps = result["next_steps"]
        self.assertTrue(any("streaming and async" in step for step in steps))

    def test_next_steps_contains_api_integration(self):
        """Test that next steps include API integration"""
        result = next_steps_planner_node({})
        steps = result["next_steps"]
        self.assertTrue(any("external APIs" in step for step in steps))

    def test_next_steps_all_strings(self):
        """Test that all next steps are strings"""
        result = next_steps_planner_node({})
        steps = result["next_steps"]
        self.assertTrue(all(isinstance(step, str) for step in steps))

    def test_next_steps_no_empty_strings(self):
        """Test that no next steps are empty strings"""
        result = next_steps_planner_node({})
        steps = result["next_steps"]
        self.assertTrue(all(len(step) > 0 for step in steps))

    def test_next_steps_with_existing_state(self):
        """Test that node works with pre-existing state"""
        state = {
            "topics_covered": [],
            "skills_acquired": [],
            "next_steps": ["existing step"],
            "final_thoughts": ""
        }
        result = next_steps_planner_node(state)
        self.assertEqual(len(result["next_steps"]), 8)

    def test_next_steps_idempotent(self):
        """Test that calling the function multiple times produces same result"""
        result1 = next_steps_planner_node({})
        result2 = next_steps_planner_node({})
        self.assertEqual(result1["next_steps"], result2["next_steps"])


class TestReflectionNode(unittest.TestCase):
    """Test suite for reflection_node function"""

    def test_reflection_returns_string(self):
        """Test that reflection_node returns a string"""
        result = reflection_node({})
        self.assertIn("final_thoughts", result)
        self.assertIsInstance(result["final_thoughts"], str)

    def test_reflection_not_empty(self):
        """Test that reflection is not empty"""
        result = reflection_node({})
        self.assertGreater(len(result["final_thoughts"]), 0)

    def test_reflection_contains_langgraph(self):
        """Test that reflection mentions LangGraph"""
        result = reflection_node({})
        self.assertIn("LangGraph", result["final_thoughts"])

    def test_reflection_contains_paradigm_shift(self):
        """Test that reflection mentions paradigm shift"""
        result = reflection_node({})
        self.assertIn("paradigm shift", result["final_thoughts"])

    def test_reflection_meaningful_length(self):
        """Test that reflection has meaningful length (at least 100 characters)"""
        result = reflection_node({})
        self.assertGreater(len(result["final_thoughts"]), 100)

    def test_reflection_with_existing_state(self):
        """Test that node works with pre-existing state"""
        state = {
            "topics_covered": [],
            "skills_acquired": [],
            "next_steps": [],
            "final_thoughts": "existing thoughts"
        }
        result = reflection_node(state)
        self.assertGreater(len(result["final_thoughts"]), 0)

    def test_reflection_idempotent(self):
        """Test that calling the function multiple times produces same result"""
        result1 = reflection_node({})
        result2 = reflection_node({})
        self.assertEqual(result1["final_thoughts"], result2["final_thoughts"])

    def test_reflection_contains_key_concepts(self):
        """Test that reflection contains key concepts like graphs, stateful, agents"""
        result = reflection_node({})
        reflection = result["final_thoughts"].lower()
        self.assertIn("graph", reflection)
        self.assertIn("stateful", reflection)


class TestSummaryState(unittest.TestCase):
    """Test suite for SummaryState TypedDict"""

    def test_state_creation_with_all_fields(self):
        """Test that SummaryState can be created with all fields"""
        state = {
            "topics_covered": ["topic1", "topic2"],
            "skills_acquired": ["skill1"],
            "next_steps": ["step1"],
            "final_thoughts": "thoughts"
        }
        # Verify it has the expected structure
        self.assertIn("topics_covered", state)
        self.assertIn("skills_acquired", state)
        self.assertIn("next_steps", state)
        self.assertIn("final_thoughts", state)

    def test_state_list_fields_are_lists(self):
        """Test that list fields in state are actually lists"""
        state = {
            "topics_covered": ["topic1"],
            "skills_acquired": ["skill1"],
            "next_steps": ["step1"],
            "final_thoughts": "thoughts"
        }
        self.assertIsInstance(state["topics_covered"], list)
        self.assertIsInstance(state["skills_acquired"], list)
        self.assertIsInstance(state["next_steps"], list)

    def test_state_final_thoughts_is_string(self):
        """Test that final_thoughts field is a string"""
        state = {
            "topics_covered": [],
            "skills_acquired": [],
            "next_steps": [],
            "final_thoughts": "thoughts"
        }
        self.assertIsInstance(state["final_thoughts"], str)


class TestGraphConstruction(unittest.TestCase):
    """Test suite for graph construction and compilation"""

    def test_graph_creation(self):
        """Test that StateGraph can be created with SummaryState"""
        builder = StateGraph(SummaryState)
        self.assertIsNotNone(builder)

    def test_graph_add_nodes(self):
        """Test that nodes can be added to the graph"""
        builder = StateGraph(SummaryState)
        builder.add_node("topics_review", topics_review_node)
        builder.add_node("skills_assessment", skills_assessment_node)
        builder.add_node("next_steps_planner", next_steps_planner_node)
        builder.add_node("reflection", reflection_node)
        # If no exception is raised, the test passes
        self.assertTrue(True)

    def test_graph_add_edges(self):
        """Test that edges can be added to the graph"""
        builder = StateGraph(SummaryState)
        builder.add_node("topics_review", topics_review_node)
        builder.add_node("skills_assessment", skills_assessment_node)

        builder.add_edge("__start__", "topics_review")
        builder.add_edge("topics_review", "skills_assessment")
        # If no exception is raised, the test passes
        self.assertTrue(True)

    def test_graph_compilation(self):
        """Test that the graph can be compiled"""
        builder = StateGraph(SummaryState)
        builder.add_node("topics_review", topics_review_node)
        builder.add_node("skills_assessment", skills_assessment_node)
        builder.add_node("next_steps_planner", next_steps_planner_node)
        builder.add_node("reflection", reflection_node)

        builder.add_edge("__start__", "topics_review")
        builder.add_edge("topics_review", "skills_assessment")
        builder.add_edge("skills_assessment", "next_steps_planner")
        builder.add_edge("next_steps_planner", "reflection")
        builder.add_edge("reflection", "__end__")

        graph = builder.compile()
        self.assertIsNotNone(graph)

    @patch('sys.stdout')
    def test_graph_execution(self, mock_stdout):
        """Test that the compiled graph can be executed"""
        builder = StateGraph(SummaryState)
        builder.add_node("topics_review", topics_review_node)
        builder.add_node("skills_assessment", skills_assessment_node)
        builder.add_node("next_steps_planner", next_steps_planner_node)
        builder.add_node("reflection", reflection_node)

        builder.add_edge("__start__", "topics_review")
        builder.add_edge("topics_review", "skills_assessment")
        builder.add_edge("skills_assessment", "next_steps_planner")
        builder.add_edge("next_steps_planner", "reflection")
        builder.add_edge("reflection", "__end__")

        graph = builder.compile()

        result = graph.invoke({
            "topics_covered": [],
            "skills_acquired": [],
            "next_steps": [],
            "final_thoughts": ""
        })

        # Verify the result has all expected fields populated
        self.assertIn("topics_covered", result)
        self.assertIn("skills_acquired", result)
        self.assertIn("next_steps", result)
        self.assertIn("final_thoughts", result)

        # Verify the lists are populated
        self.assertEqual(len(result["topics_covered"]), 10)
        self.assertEqual(len(result["skills_acquired"]), 8)
        self.assertEqual(len(result["next_steps"]), 8)
        self.assertGreater(len(result["final_thoughts"]), 0)


class TestEdgeCases(unittest.TestCase):
    """Test suite for edge cases and error scenarios"""

    def test_topics_review_with_none_state(self):
        """Test topics_review_node handles None state gracefully"""
        # Should not raise an exception
        result = topics_review_node(None)
        self.assertIn("topics_covered", result)

    def test_skills_assessment_with_none_state(self):
        """Test skills_assessment_node handles None state gracefully"""
        # Should not raise an exception
        result = skills_assessment_node(None)
        self.assertIn("skills_acquired", result)

    def test_next_steps_with_none_state(self):
        """Test next_steps_planner_node handles None state gracefully"""
        # Should not raise an exception
        result = next_steps_planner_node(None)
        self.assertIn("next_steps", result)

    def test_reflection_with_none_state(self):
        """Test reflection_node handles None state gracefully"""
        # Should not raise an exception
        result = reflection_node(None)
        self.assertIn("final_thoughts", result)

    def test_empty_state_initialization(self):
        """Test that graph execution works with completely empty initial state"""
        builder = StateGraph(SummaryState)
        builder.add_node("topics_review", topics_review_node)
        builder.add_edge("__start__", "topics_review")
        builder.add_edge("topics_review", "__end__")

        graph = builder.compile()
        result = graph.invoke({
            "topics_covered": [],
            "skills_acquired": [],
            "next_steps": [],
            "final_thoughts": ""
        })

        self.assertIsNotNone(result)

    def test_node_output_format_consistency(self):
        """Test that all nodes return dictionaries with expected keys"""
        nodes = [
            (topics_review_node, "topics_covered"),
            (skills_assessment_node, "skills_acquired"),
            (next_steps_planner_node, "next_steps"),
            (reflection_node, "final_thoughts")
        ]

        for node_func, expected_key in nodes:
            result = node_func({})
            self.assertIsInstance(result, dict)
            self.assertIn(expected_key, result)

    def test_multiple_graph_executions(self):
        """Test that graph can be executed multiple times with consistent results"""
        builder = StateGraph(SummaryState)
        builder.add_node("topics_review", topics_review_node)
        builder.add_node("skills_assessment", skills_assessment_node)
        builder.add_edge("__start__", "topics_review")
        builder.add_edge("topics_review", "skills_assessment")
        builder.add_edge("skills_assessment", "__end__")

        graph = builder.compile()

        result1 = graph.invoke({
            "topics_covered": [],
            "skills_acquired": [],
            "next_steps": [],
            "final_thoughts": ""
        })

        result2 = graph.invoke({
            "topics_covered": [],
            "skills_acquired": [],
            "next_steps": [],
            "final_thoughts": ""
        })

        # Results should be identical
        self.assertEqual(result1["topics_covered"], result2["topics_covered"])
        self.assertEqual(result1["skills_acquired"], result2["skills_acquired"])


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete summary system"""

    @patch('sys.stdout')
    def test_full_summary_system_execution(self, mock_stdout):
        """Test the complete summary system from start to end"""
        builder = StateGraph(SummaryState)

        # Add all nodes
        builder.add_node("topics_review", topics_review_node)
        builder.add_node("skills_assessment", skills_assessment_node)
        builder.add_node("next_steps_planner", next_steps_planner_node)
        builder.add_node("reflection", reflection_node)

        # Set up the complete flow
        builder.add_edge("__start__", "topics_review")
        builder.add_edge("topics_review", "skills_assessment")
        builder.add_edge("skills_assessment", "next_steps_planner")
        builder.add_edge("next_steps_planner", "reflection")
        builder.add_edge("reflection", "__end__")

        # Compile and execute
        summary_system = builder.compile()
        result = summary_system.invoke({
            "topics_covered": [],
            "skills_acquired": [],
            "next_steps": [],
            "final_thoughts": ""
        })

        # Comprehensive verification
        self.assertEqual(len(result["topics_covered"]), 10)
        self.assertEqual(len(result["skills_acquired"]), 8)
        self.assertEqual(len(result["next_steps"]), 8)
        self.assertGreater(len(result["final_thoughts"]), 100)

        # Verify specific content
        self.assertIn("LangGraph fundamentals and architecture", result["topics_covered"])
        self.assertIn("Designing stateful graph applications", result["skills_acquired"])
        self.assertTrue(any("streaming" in step for step in result["next_steps"]))
        self.assertIn("LangGraph", result["final_thoughts"])

    def test_state_accumulation_through_graph(self):
        """Test that state properly accumulates as it flows through the graph"""
        builder = StateGraph(SummaryState)

        builder.add_node("topics_review", topics_review_node)
        builder.add_node("skills_assessment", skills_assessment_node)
        builder.add_node("next_steps_planner", next_steps_planner_node)

        builder.add_edge("__start__", "topics_review")
        builder.add_edge("topics_review", "skills_assessment")
        builder.add_edge("skills_assessment", "next_steps_planner")
        builder.add_edge("next_steps_planner", "__end__")

        graph = builder.compile()
        result = graph.invoke({
            "topics_covered": [],
            "skills_acquired": [],
            "next_steps": [],
            "final_thoughts": ""
        })

        # All three node outputs should be in the final state
        self.assertIsNotNone(result["topics_covered"])
        self.assertIsNotNone(result["skills_acquired"])
        self.assertIsNotNone(result["next_steps"])


if __name__ == "__main__":
    unittest.main()