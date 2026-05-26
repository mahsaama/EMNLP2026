####### 5-POINT LIKERT EVALUATION #######

SYSTEM_PROMPT_FACTUALITY_5LIKERT = """
You are an evaluator assessing the factual correctness of an AI-generated response to a user query.

Evaluate:
- Is the response factually correct and free from hallucinations or false claims?
- Is the information up-to-date and not outdated when recency matters?

Return JSON:

{{
"score": 1-5,
"reasoning": "<1-2 sentence explanation>"
}}

Scoring guide:
1 = Mostly incorrect or clearly hallucinated; core claims are wrong
2 = More incorrect than correct; contains significant factual errors that undermine the answer, even if some parts are right
3 = Mixed accuracy; contains both correct and incorrect claims of similar importance
4 = Mostly correct; minor inaccuracies or slightly outdated details that do not change the overall answer
5 = Fully correct, precise, and up-to-date; no meaningful errors

Before scoring, consider the query type:
- For creative queries, interpret factuality as internal consistency rather than real-world truth.
"""

SYSTEM_PROMPT_COMPLETENESS_5LIKERT = """
You are an evaluator assessing the completeness of an AI-generated response to a user query.

Evaluate:
- Does the response fully address and cover all parts of the user’s question?

Return JSON:

{{
  "score": 1-5,
  "reasoning": "<1-2 sentence explanation>"
}}

Scoring guide:
1 = Very incomplete; misses most parts of the question or fails to address the main request
2 = Partially incomplete; addresses some parts but omits major components of the question
3 = Moderately complete; covers the main request but misses some secondary aspects or details
4 = Mostly complete; addresses nearly all parts with only minor omissions
5 = Fully complete; covers all aspects of the question thoroughly

Before scoring, consider the query type:
- For open-ended queries, interpret completeness as reasonable coverage of key aspects, not exhaustiveness.
"""

SYSTEM_PROMPT_RELEVANCE_5LIKERT = """
You are an evaluator assessing how relevant an AI-generated response is to a user query.

Evaluate:
- Does the response directly address the user's question or intent?
- Is the response concise, to the point, and free from off-topic or unnecessary information?

Return JSON:

{{
"score": 1-5,
"reasoning": "<1-2 sentence explanation>"
}}

Scoring guide:
1 = Irrelevant; does not address the user’s question or intent at all
2 = Weakly relevant; touches on the topic but largely misses the user’s intent or includes substantial off-topic content
3 = Partially relevant; addresses the main intent but includes noticeable irrelevance or digressions
4 = Mostly relevant; well-aligned with the intent with only minor off-topic or unnecessary details
5 = Fully relevant; directly and precisely addresses the user’s intent with no unnecessary content
"""

USER_PROMPT_5LIKERT = """
Evaluate the following.

User Query:
{user_query}

AI Response:
{response}

Return ONLY valid JSON in this exact format:
{{
"score": <integer 1-5>,
"reasoning": "<string>"
}}

Rules:
- Do not include any text outside the JSON
- Do not add explanations before or after
- Ensure the JSON is valid
"""

SYSTEM_PROMPT_USER_WEB_QUERY_TYPE = """
You are an evaluator classifying the intent of a query.

Classify the query into exactly ONE of the following categories:

- "informational":
  The query is primarily seeking information, explanations, facts, answers, or learning content about a topic.

- "navigational":
  The query is primarily intended to find or access a specific website, webpage, app, platform, or online resource.

- "transactional":
  The query is primarily intended to perform an action using an online service, platform, or tool, such as buying, booking, downloading, signing up, or creating something.

- "commercial":
  The query is primarily intended to research or compare products/services with potential purchase intent, but without explicitly attempting to complete a transaction.


Guidelines:
- Choose the SINGLE best category
- Focus on the primary intent of the query

Examples:

Query: "I am shopping for a flight from KUL to Guadalajara..."
→ transactional

Query: "Open the Stanford CS229 course website"
→ navigational

Query: "How can I reset my Instagram password if I don’t have access to my original email account?"
→ informational

Query: "best noise cancelling headphones"
→ commercial

Return ONLY valid JSON:

{{
  "type": "transactional" | "navigational" | "informational" | "commercial",
  "reasoning": "<1-2 sentence explanation>"
}}

Rules:
- Output must be valid JSON
- Do not include any extra text
"""

USER_PROMPT_USER_WEB_QUERY_TYPE = """
Classify the following query into exactly one intent type.

Query:
{user_query}

Return ONLY valid JSON:
{{
  "type": "transactional" | "navigational" | "informational" | "commercial,
  "reasoning": "<string>"
}}

Rules:
- Choose exactly one type
- Focus on the primary intent
- Output valid JSON only
"""

SYSTEM_PROMPT_RESP_SYNT = """
You are an NLI (Natural Language Inference) judge.

Given a response chunk and source content, determine the relationship between them.

Labels:
- entailment: the source content supports or expresses the same meaning as the response chunk
- contradiction: the source content conflicts with the response chunk on a meaningful point
- neutral: the source content does not provide enough information to support or contradict the response chunk

Rules:
- Treat the response as a chunk or partial segment, not necessarily a complete standalone answer.
- Evaluate only the claims explicitly present in the response chunk.
- Use only the provided source content.
- Do not use external knowledge.
- Base your decision on semantic meaning, not exact wording.
- If the chunk contains multiple claims and only some are supported, prefer neutral unless there is a clear contradiction.
- Use contradiction only when the source clearly conflicts with the response chunk.
- Do not penalize the chunk for being incomplete, abbreviated, or lacking surrounding context.

Output JSON only:
{{
  "label": "entailment" | "contradiction" | "neutral",
  "reason": "<1-2 sentence explanation>",
  "score": 1-5
}}
"""

USER_PROMPT_RESP_SYNT = """
Response Chunk:
{response_text}

Source Content:
{source}

Determine whether the response chunk is entailed by, contradicts, or is neutral with respect to the source content.

Return ONLY valid JSON:
{{
  "label": "entailment|contradiction|neutral",
  "reason": "<string>",
  "score": <integer 1-5>
}}
"""

SYSTEM_PROMPT_QUERY_REASON = """
You are an expert evaluator of conversational search behavior, specializing in query reformulation.

Your task is to label the relationship between specified query transitions based on whether the reformulated query improves upon the original query in 1 of the 2 following ways:

You must choose exactly one of the following categories:

1. Query Rewriting
- The query is reformulated into a clearer, self-contained, or less ambiguous form.
- Often resolves ambiguity or rewrites the query to better reflect the user’s intent.

2. Query Expansion
- The query is augmented with additional terms or context.
- Adds missing details, constraints, or related concepts to better specify the information need.

3. Hybrid
- Combines both rewriting and expansion.
- The query is both clarified/rephrased AND enriched with new information.

4. Other
- The refomulated query is neither more clarified nor enriched with new information.
- The reformulated query does not constitute a clear improvement over the original query. So, it cannot be labeled either as query rewriting or query expansion.

Instructions:
- You are given:
  - The original user query (ID: U)
  - A set of web queries with IDs like 1.1, 2.1, etc.
  - A list of transition pairs to classify
  - Thinking traces explaining why the next query was issued
- For each listed transition (from -> to), assign exactly one label.
- Base your decision only on how the "to" query is reformulated relative to the "from" query.
- Treat U as the original user query text.
- If multiple categories seem applicable, select the dominant reformulation strategy.
- Provide a short reasoning (1–2 sentences) grounded in these definitions.
- Return every listed transition exactly once, and do not add extra transitions.

Output format (STRICT JSON):
{{
  "transitions": [
    {{
      "from": "U",
      "to": "1.1",
      "label": "Query Rewriting | Query Expansion | Hybrid | Other",
      "reasoning": "1-2 sentence explanation"
    }}
  ]
}}
"""

USER_PROMPT_QUERY_REASON = """
Classify the listed transitions using conversational search query reformulation terminology.

Example:

User Query (U):
Best laptops for programming

Web Queries:
(1.1) best laptops for programmers
(2.1) best lightweight laptops for programming students
(2.2) macbook air m3 student programming battery life

Thinking Traces:
(1.1) "I should rephrase this into a direct benchmark-style web query."
(2.1) "I want results tailored for students and portability, so I will add lightweight and student-related constraints."
(2.2) "I should also check a concrete model line and include battery-life angle for students."

Transitions to classify:
(U -> 1.1)
(1.1 -> 2.1)
(1.1 -> 2.2)

Output:
{{
  "transitions": [
    {{
      "from": "U",
      "to": "1.1",
      "label": "Query Rewriting",
      "reasoning": "The first web query is a clarified, self-contained rewrite of the user request with minimal new constraints."
    }},
    {{
      "from": "1.1",
      "to": "2.1",
      "label": "Query Expansion",
      "reasoning": "The second query adds new constraints and contextual attributes ('lightweight' and 'students') to better specify the information need."
    }},
    {{
      "from": "1.1",
      "to": "2.2",
      "label": "Hybrid",
      "reasoning": "The query shifts to a specific product family while adding several new constraints (student use and battery life), combining rewriting and expansion."
    }}
  ]
}}

Now classify this:

User Query (U):
{user_query}

Web Queries:
{web_queries}

Thinking Traces:
{thinking_traces}

Transitions to classify:
{transition_candidates}

Return ONLY the JSON.
"""

SYSTEM_PROMPT_QUERY_REASON_VALIDATOR = """
You are an expert evaluator of conversational search behavior, specializing in query reformulation.

Your task is to label the relationship between specified query transitions based on whether the reformulated query improves upon the original query in 1 of the 2 following ways:

You must choose exactly one of the following categories:

1. Query Rewriting
- The query is reformulated into a clearer, self-contained, or less ambiguous form.
- Often resolves ambiguity or rewrites the query to better reflect the user’s intent.

2. Query Expansion
- The query is augmented with additional terms or context.
- Adds missing details, constraints, or related concepts to better specify the information need.

3. Hybrid
- Combines both rewriting and expansion.
- The query is both clarified/rephrased AND enriched with new information.

4. Other
- The refomulated query is neither more clarified nor enriched with new information.
- The reformulated query does not constitute a clear improvement over the original query. So, it cannot be labeled either as query rewriting or query expansion.

Instructions:
- You are given:
  - The original user query (ID: U)
  - A set of web queries with IDs like 1.1, 2.1, etc.
  - A list of transition pairs to classify
- For each listed transition (from -> to), assign exactly one label.
- Base your decision only on how the "to" query is reformulated relative to the "from" query.
- Treat U as the original user query text.
- If multiple categories seem applicable, select the dominant reformulation strategy.
- Provide a short reasoning (1–2 sentences) grounded in these definitions.
- Return every listed transition exactly once, and do not add extra transitions.

Output format (STRICT JSON):
{{
  "transitions": [
    {{
      "from": "U",
      "to": "1.1",
      "label": "Query Rewriting | Query Expansion | Hybrid | Other",
      "reasoning": "1-2 sentence explanation"
    }}
  ]
}}
"""

USER_PROMPT_QUERY_REASON_VALIDATOR = """
Classify the listed transitions using conversational search query reformulation terminology.

Example:

User Query (U):
Best laptops for programming

Web Queries:
(1.1) best laptops for programmers
(2.1) best lightweight laptops for programming students
(2.2) macbook air m3 student programming battery life

Transitions to classify:
(U -> 1.1)
(1.1 -> 2.1)
(1.1 -> 2.2)

Output:
{{
  "transitions": [
    {{
      "from": "U",
      "to": "1.1",
      "label": "Query Rewriting",
      "reasoning": "The first web query is a clarified, self-contained rewrite of the user request with minimal new constraints."
    }},
    {{
      "from": "1.1",
      "to": "2.1",
      "label": "Query Expansion",
      "reasoning": "The second query adds new constraints and contextual attributes ('lightweight' and 'students') to better specify the information need."
    }},
    {{
      "from": "1.1",
      "to": "2.2",
      "label": "Hybrid",
      "reasoning": "The query shifts to a specific product family while adding several new constraints (student use and battery life), combining rewriting and expansion."
    }}
  ]
}}

Now classify this:

User Query (U):
{user_query}

Web Queries:
{web_queries}

Transitions to classify:
{transition_candidates}

Return ONLY the JSON.
"""

SYSTEM_PROMPT_CLAIM_EXTRACTION = """
You are an expert claim extraction system.

Your task is to identify and extract claims from text.

Definition of a claim:
A claim is any assertion, proposition, statement, opinion, prediction, or description that could be evaluated, supported, contradicted, or discussed.

Rules:
- Extract all meaningful claims expressed in the text.
- Rewrite claims as standalone declarative sentences.
- Resolve pronouns and references where possible.
- Split compound sentences into atomic claims whenever appropriate.
- Preserve the original meaning, including:
  - negation
  - modality
  - uncertainty
  - comparisons
  - quantities
  - temporal information
- Do not infer unstated information.
- Avoid duplicate claims.
- Keep claims concise and self-contained.

Output requirements:
- Return ONLY a valid JSON array of strings.
- Do not include explanations or additional text.
"""

USER_PROMPT_CLAIM_EXTRACTION = """
Extract all claims from the following text.

Text:
{text}
"""