from typing import Callable, Dict, List

import pandas as pd
from joblib import Memory
from tqdm.auto import tqdm

from external_weaviate.llm import ChatLLM

memory = Memory("./my_cache_dir", verbose=0)
cache = memory.cache

tqdm.pandas()


# -- Generate Summary for a List of Paragraphs --
@cache
def generate_summary(paragraphs: List[str], llm: ChatLLM) -> str:
    """
    Given a list of paragraphs, generate a concise summary using a language model.
    """
    if isinstance(paragraphs, str):
        paragraphs = [paragraphs]
    text = "\n\n".join(paragraphs)
    prompt = f"Summarize the following section in 2-3 sentences:\n\n{text}"

    summary = llm.call(prompt, temperature=0.5)
    return summary

@cache
def get_children(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    returns a dictionary of each node's id to all children of that node (no order guaranteed)
    """
    df = df.set_index("uid")
    child_lists: Dict[str, List[str]] = {uid: [] for uid in df.index}
    child_df = df[df.parent.notna()]
    for uid, row in tqdm(child_df.iterrows()):
        child_lists[row["parent"]].append(uid)
    return child_lists


def get_recursive_summary(
    uid: str,
    children: Dict,
    content: Dict,
    summarizer: Callable[[str], str],
    summary_cache: Dict[str, str],
) -> str:
    """
    Summarizes a node's contents with all its children's contents. Uses a cache to only summarize each node once.
    """
    if uid in summary_cache:
        return summary_cache[uid]
    else:
        combined_content = "\n".join(
            [content[uid]]
            + [
                get_recursive_summary(
                    uid=cuid,
                    children=children,
                    content=content,
                    summarizer=summarizer,
                    summary_cache=summary_cache,
                )
                for cuid in children[uid]
            ]
        )
        summary = summarizer(combined_content)
        summary_cache[uid] = summary
        return summary

@cache
def summarize_all(df: pd.DataFrame, summarizer: Callable[[str], str]) -> pd.DataFrame:
    # summarize all children of each node
    children: Dict = get_children(df)
    content = df.set_index("uid").content
    summaries = {}
    for uid in tqdm( df.uid, initial=0, total=len(df.uid)):
        print(f"{len(summaries)=}/{len(df.uid)}")
        summaries.get(
            uid,
            get_recursive_summary(
                uid=uid,
                children=children,
                content=content,
                summarizer=summarizer,
                summary_cache=summaries,
            ),
        )
    return df.uid.progress_apply(lambda uid: summaries[uid])

