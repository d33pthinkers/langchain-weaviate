import os
import random
import shutil
import unittest
from joblib import Memory
from openai import BaseModel

from external_weaviate.llm import ChatLLM

# Setup a temporary cache directory for testing
CACHE_DIR = "./test_cache"
memory = Memory(CACHE_DIR, verbose=0)

# ------------------------
# 1. Static Function

call_counter_static = {"count": 0}

@memory.cache
def static_cached_fn(x):
    call_counter_static["count"] += 1
    return f"computed-{x}"

@memory.cache
def static_calling_fn(x, *args, **kwargs):
    call_counter_static["count"] += 1
    return f"computed-{x(*args, **kwargs)}"
# ------------------------
# 2. Instance Method (with __getstate__)
class StatefulObject:
    call_counter = {"count": 0}

    def __init__(self, scale):
        self.scale = scale

    def __getstate__(self):
        return {"scale": self.scale}

    def __setstate__(self, state):
        self.__dict__.update(state)

    def multiply(self, x):
        return self._cached_multiply(x)

    @memory.cache
    def _cached_multiply(self, x):
        StatefulObject.call_counter["count"] += 1
        return x * self.scale

# ------------------------
# 3. Instance Method (expected to fail)
class FailingObject:
    def __init__(self, factor):
        self.factor = factor

    @memory.cache
    def multiply(self, x):
        return x * self.factor

# ------------------------
# Pydantic classs as input
class MyPydanticClass(BaseModel):
    config1: str
    config2: str
class MyCallablePydanticClass(BaseModel):
    config1: str
    config2: str
    def __call__(self, *args, **kwargs):
        return random.randint(0, 10000)
# ------------------------
# UNIT TESTS
class TestJoblibCaching(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(CACHE_DIR):
            shutil.rmtree(CACHE_DIR)

        call_counter_static = {"count": 0}

    def test_static_function_caching(self):
        call_counter_static["count"] = 0
        result1 = static_cached_fn(10)
        result2 = static_cached_fn(10)
        self.assertEqual(result1, result2)
        self.assertEqual(call_counter_static["count"], 1)
    def test_static_function_with_pydantic_inputs(self):
        call_counter_static["count"] = 0

        result1 = static_cached_fn(MyPydanticClass(config1="config1", config2="config2"))
        result2 = static_cached_fn(MyPydanticClass(config1="config1", config2="config2"))
        self.assertEqual(result1, result2)
        self.assertEqual(call_counter_static["count"], 1)
    def test_static_function_with_changing_pydantic_inputs(self):
        call_counter_static["count"] = 0

        result1 = static_cached_fn(MyPydanticClass(config1="config1", config2="config2"))
        result2 = static_cached_fn(MyPydanticClass(config1="config3", config2="config2"))
        self.assertNotEquals(result1, result2)
        self.assertEqual(call_counter_static["count"], 2)
    def test_static_function_with_function_inputs(self):
        call_counter_static["count"] = 0

        result1 = static_cached_fn(MyCallablePydanticClass(config1="config1", config2="config2").__call__)
        result2 = static_cached_fn(MyCallablePydanticClass(config1="config1", config2="config2").__call__)
        self.assertEqual(result1, result2)
        self.assertEqual(call_counter_static["count"], 1)
    def test_static_function_with_callable_inputs(self):
        call_counter_static["count"] = 0

        result1 = static_calling_fn(MyCallablePydanticClass(config1="config1", config2="config2"))
        result2 = static_calling_fn(MyCallablePydanticClass(config1="config1", config2="config2"))
        self.assertEqual(result1, result2)
        self.assertEqual(call_counter_static["count"], 1)
    def test_static_function_with_different_callable_inputs(self):
        call_counter_static["count"] = 0

        result1 = static_calling_fn(MyCallablePydanticClass(config1="config1", config2="config2"))
        result2 = static_calling_fn(MyCallablePydanticClass(config1="config3", config2="config2"))
        self.assertNotEquals(result1, result2)
        self.assertEqual(call_counter_static["count"], 2)

    def test_instance_method_with_getstate(self):
        StatefulObject.call_counter["count"] = 0
        obj = StatefulObject(scale=3)
        result1 = obj.multiply(5)
        result2 = obj.multiply(5)
        self.assertEqual(result1, result2)
        self.assertEqual(StatefulObject.call_counter["count"], 1)

    def test_instance_method_with_matching_llms(self):
        StatefulObject.call_counter["count"] = 0
        LLM_URL = os.getenv("LLM_URL")
        SUMMARIZER_MODEL = os.getenv("SUMMARIZER_MODEL")
        llm1 = ChatLLM(
            model=SUMMARIZER_MODEL,
            base_url=LLM_URL,
        )
        llm2 = ChatLLM(
            model=SUMMARIZER_MODEL,
            base_url=LLM_URL,
        )
        result1 = static_calling_fn(llm1, "give me 5", temperature=1)
        result2 = static_calling_fn(llm2, "give me 5", temperature=1)
        self.assertEqual(result1, result2), f"{result1} != {result2}"
        self.assertEqual(call_counter_static["count"], 1)

    def test_instance_method_with_different_llms(self):
        StatefulObject.call_counter["count"] = 0
        LLM_URL = os.getenv("LLM_URL")
        SUMMARIZER_MODEL = os.getenv("SUMMARIZER_MODEL")
        llm1 = ChatLLM(
            model=SUMMARIZER_MODEL,
            base_url=LLM_URL,
            temperature=0.4
        )
        llm2 = ChatLLM(
            model=SUMMARIZER_MODEL,
            base_url=LLM_URL,
            temperature=0.5
        )
        result1 = static_calling_fn(llm1, "give me 5")
        result2 = static_calling_fn(llm2, "give me 5")
        self.assertNotEquals(result1, result2), f"{result1} != {result2}"
        self.assertEqual(2, call_counter_static["count"])

    def test_instance_method_without_getstate_should_fail(self):
        obj = FailingObject(factor=2)
        with self.assertRaises(Exception):
            _ = obj.multiply(10)

unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(TestJoblibCaching))
