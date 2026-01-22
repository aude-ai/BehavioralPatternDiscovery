"""
Prompt Templates for Pattern Naming

Contains all static prompt text content, templates, and descriptions.
Separated from PromptBuilder logic for better organization and maintainability.

This file contains:
- SYSTEM_CONTEXT: What VAEs are and what the task is
- DATA_SOURCE_DESCRIPTIONS: Info about each data source type
- TEAM_CONTEXT: Engineering team dynamics
- DISENTANGLEMENT_CONTEXT: How to interpret disentangled patterns
- ABSTRACTION_LEVELS: Abstraction degree info per level
- LEVEL_CONTEXTS: Level-specific guidance (first, middle, final_encoder, unified)
- NAMING_GUIDELINES_TEMPLATE: Rules for good pattern names
- OUTPUT_FORMAT_TEMPLATE: Expected JSON structure
"""

# =============================================================================
# SYSTEM CONTEXT - What VAEs are and what the task is
# =============================================================================

SYSTEM_CONTEXT = """
## Your Role

You are an expert at analyzing machine learning model outputs to identify observable
behavioral patterns in how engineers communicate and collaborate.

You are examining **latent dimensions** from a Variational Autoencoder (VAE) that has been
trained on software engineer communications. Each dimension represents a **distinct
behavioral pattern** in how engineers operate as team members.

## Your Task

For each dimension/pattern:
1. Examine the messages that most strongly activate it
2. Identify what **behavioral pattern** that dimension captures - focus on HOW the person
   communicates and collaborates, NOT what topics/projects they work on
3. Provide a concise name and brief description

## Critical Distinction

**FOCUS ON**: How the engineer behaves as a team member
- Communication style (verbose/concise, proactive/reactive, formal/casual)
- Collaboration approach (helps others, seeks input, works independently)
- Quality orientation (thorough, quick, detail-focused)
- Problem-solving style (systematic, intuitive, collaborative)
- Feedback patterns (constructive, critical, encouraging)

**AVOID**: What topics or projects they work on
- "Testing" is a topic, not a behavior
- "Database work" is a domain, not a behavior
- "Bug fixes" is a task type, not a behavior

The same engineer could work on testing, databases, or bug fixes while exhibiting
different BEHAVIORS: thorough analysis, quick responses, detailed explanations, etc.

## How Patterns Will Be Used

Engineering managers will use these patterns to understand their team:
- How does each person contribute to team dynamics?
- What communication strengths and gaps exist?
- How do different team members collaborate?
- Where can individuals grow as team members?

The patterns should help a manager understand HOW someone operates on a team,
not WHAT projects they happen to work on.
"""

# =============================================================================
# DATA SOURCE DESCRIPTIONS - Info about each platform
# =============================================================================

DATA_SOURCE_DESCRIPTIONS = {
    "github": {
        "name": "GitHub",
        "types": [
            "Pull request descriptions and titles",
            "Code review comments and feedback",
            "Commit messages describing changes",
            "Issue discussions and bug reports",
            "Review approvals and change requests",
        ],
    },
    "slack": {
        "name": "Slack",
        "types": [
            "Team chat messages",
            "Technical questions and answers",
            "Coordination and status updates",
            "Informal discussions",
            "Direct messages (if included)",
        ],
    },
    "jira": {
        "name": "Jira",
        "types": [
            "Ticket descriptions and summaries",
            "Comment threads on issues",
            "Status updates and transitions",
            "Sprint planning notes",
            "Bug reports and feature requests",
        ],
    },
    "confluence": {
        "name": "Confluence",
        "types": [
            "Technical documentation",
            "Project wikis and specs",
            "Meeting notes",
            "Design documents",
            "Onboarding guides",
        ],
    },
    "trello": {
        "name": "Trello",
        "types": [
            "Card descriptions",
            "Checklist items",
            "Card comments",
            "Board activity",
        ],
    },
}

DATA_SOURCE_HEADER = """
## Data Sources

The VAE was trained on engineer communications from the following platforms:
"""

MESSAGE_CHARACTERISTICS = """
## Message Characteristics

Messages vary in:
- **Length**: From brief commit messages to detailed documentation
- **Formality**: From casual Slack chat to formal specifications
- **Technical depth**: From high-level summaries to implementation details
- **Audience**: From self-notes to team-wide announcements
"""

# =============================================================================
# WORD ATTRIBUTION CONTEXT - Explains word-level attributions in examples
# =============================================================================

WORD_ATTRIBUTION_CONTEXT = """
## Word-Level Attribution

Each pattern includes **aggregated word attribution scores** showing which specific
words most strongly influence activation for that pattern across all top messages.

### How to Interpret Word Attributions

- **Positive mean_delta** (> 0): Word INCREASES the pattern score - characteristic vocabulary that ACTIVATES this pattern
- **Negative mean_delta** (< 0): Word DECREASES the pattern score - suppresses this pattern or associated with other patterns

### Attribution Data

- **word**: The word analyzed
- **mean_delta**: Average score change when the word is removed (positive = word contributes to pattern)
- **occurrences**: How many top messages contain this word

Words with high positive mean_delta appearing frequently are the most reliable indicators
of what linguistic signals characterize this behavioral pattern.

### Using Word Attributions for Naming

The word attributions reveal the **vocabulary fingerprint** of each pattern:
- Look for thematic clusters in high-scoring words (e.g., "refactored", "simplified", "cleanup")
- Consider what behavior would produce this vocabulary
- The top words shown are those most characteristic of this pattern

**Example interpretation:**
- High-scoring words: "refactored", "simplified", "consolidated" -> Pattern captures code improvement behavior
- This vocabulary indicates someone who focuses on cleaning up and improving existing code

This helps distinguish patterns that might have similar messages but different underlying behaviors.
"""

# =============================================================================
# TEAM CONTEXT - Engineering team dynamics
# =============================================================================

TEAM_CONTEXT = """
## Engineering Team Context

These communications come from **software engineering teams**. The patterns may capture:

### Communication Styles
- How engineers explain technical concepts to others
- Tone and formality levels (casual vs. professional)
- Question-asking behaviors vs. answer-providing behaviors
- Verbosity and detail levels in explanations

### Technical Behaviors
- Code review thoroughness and feedback quality
- Documentation practices and attention to detail
- Problem-solving approaches (systematic vs. intuitive)
- Technical depth and breadth of knowledge

### Collaboration Patterns
- Leadership and mentorship indicators
- Cross-team coordination behaviors
- Knowledge sharing and teaching patterns
- Responsiveness and engagement levels

### Role-Indicative Patterns
- **Manager/Lead patterns**: Delegation, status requests, planning, unblocking
- **Senior engineer patterns**: Mentoring, architectural discussions, code quality focus
- **Individual contributor patterns**: Implementation details, questions, learning behaviors
- **Cross-functional patterns**: Translation between technical and non-technical contexts
"""

# =============================================================================
# DISENTANGLEMENT CONTEXT - How to interpret patterns (CRITICAL)
# =============================================================================

DISENTANGLEMENT_CONTEXT = """
## Understanding Disentangled Patterns

The VAE has been trained to produce **disentangled representations**. This is critical
for interpreting the patterns:

### What Disentanglement Means

1. **Each dimension captures a DISTINCT pattern** - The model is regularized to ensure
   different dimensions encode different aspects of behavior, minimizing redundancy.

2. **Patterns are largely independent** - High activation on one dimension does NOT
   predict activation on another dimension. They measure orthogonal aspects.

3. **No two patterns should mean the same thing** - If you find yourself giving two
   patterns the same name, you're missing something.

### How to Handle Seemingly Similar Patterns

If two patterns appear to capture the same thing, **look deeper** for the distinction:

- **WHAT vs. HOW**: One might capture what is communicated, another how it's communicated
- **Frequency vs. Depth**: One might capture how often, another how thoroughly
- **Context differences**: Same behavior but in different contexts (code review vs. chat)
- **Subtle tone differences**: Both "helpful" but one is "proactively helpful" vs. "reactively helpful"
- **Scope differences**: One is about individual messages, another about conversation patterns

### The Key Insight

**The contrast between seemingly similar patterns often reveals their true distinction.**

When two patterns have overlapping top messages, ask:
- What messages appear in one but NOT the other?
- What's different about the activation scores?
- Could they represent the same behavior at different intensities?
- Could they represent the same behavior in different contexts?

### Naming Implications

- Give each pattern a **unique, specific name**
- Avoid generic names that could apply to multiple patterns
- If stuck, describe what makes THIS pattern different from others
- Use qualifying words: "Proactive" vs. "Reactive", "Brief" vs. "Detailed", "Technical" vs. "Conceptual"
"""

# =============================================================================
# ABSTRACTION LEVELS - Describes the abstraction degree for each level
# =============================================================================

# Maps level names (from model.yaml encoder.hierarchical.levels) to abstraction info.
# When levels are added/changed in config, update this mapping.
# The key is the level name, value is a dict with:
#   - "degree": Short label (e.g., "fine-grained", "thematic")
#   - "description": What patterns at this level represent
#   - "detail": Extended explanation
#   - "examples": Example pattern types at this abstraction

ABSTRACTION_LEVELS = {
    "bottom": {
        "degree": "atomic",
        "description": "specific communication actions",
        "detail": (
            "These are the most granular behavioral patterns - specific actions "
            "observable in individual messages. Focus on HOW someone communicates, "
            "not what they're communicating about."
        ),
        "examples": [
            "Asking clarifying questions before acting",
            "Providing step-by-step explanations",
            "Acknowledging others' contributions",
            "Giving specific, actionable feedback",
            "Summarizing decisions for the record",
        ],
    },
    "mid": {
        "degree": "composite",
        "description": "behavioral tendencies from combined actions",
        "detail": (
            "These patterns represent consistent behavioral tendencies that emerge "
            "when multiple atomic actions appear together. They describe HOW someone "
            "typically operates, not what domain they work in."
        ),
        "examples": [
            "Thorough analysis before responding",
            "Proactive status communication",
            "Collaborative problem-solving approach",
            "Constructive feedback delivery",
            "Clear escalation of blockers",
        ],
    },
    "top": {
        "degree": "characteristic",
        "description": "higher-order behavioral characteristics",
        "detail": (
            "These patterns capture broader behavioral characteristics that define "
            "how someone functions as a team member. They should describe observable "
            "patterns in how someone collaborates and communicates."
        ),
        "examples": [
            "Cross-team coordination style",
            "Knowledge sharing orientation",
            "Independent vs. collaborative working",
            "Responsive support to teammates",
            "Systematic vs. intuitive approach",
        ],
    },
    "unified": {
        "degree": "evaluative",
        "description": "broad behavioral dimensions for team member evaluation",
        "detail": (
            "The most abstract behavioral level - synthesizes patterns from all encoders. "
            "Since each encoder independently discovers patterns, some may have found similar "
            "or overlapping patterns. The unified level combines these into broad evaluation "
            "dimensions. These are the TOP-LEVEL dimensions a manager uses to evaluate team "
            "members. They must be BROAD enough that ANY engineer can be scored on them, "
            "regardless of their technical specialty. Think: performance review categories, not job roles."
        ),
        "examples": [
            "Communication Clarity",
            "Collaborative Problem-Solving",
            "Proactive Initiative",
            "Thoroughness & Attention to Detail",
            "Responsiveness to Team Needs",
            "Knowledge Sharing",
            "Constructive Feedback",
            "Ownership & Follow-Through",
        ],
    },
}


def get_abstraction_for_level(level_name: str) -> dict:
    """
    Get abstraction info for a level name.

    Falls back to generic description if level not in mapping.
    """
    if level_name in ABSTRACTION_LEVELS:
        return ABSTRACTION_LEVELS[level_name]

    # Fallback for unknown levels
    return {
        "degree": "intermediate",
        "description": f"patterns at the {level_name} level",
        "detail": f"Patterns at this level represent behaviors captured by the {level_name} layer.",
        "examples": [],
    }


# =============================================================================
# LEVEL CONTEXTS - Level-specific guidance
# =============================================================================

LEVEL_CONTEXT_FIRST = """
## Current Level: {level_name}

**Abstraction Level: {current_abstraction_degree}** - {current_abstraction_description}

You are naming patterns at the **first/lowest level** of the hierarchy for **{encoder_name}**.

### What This Level Captures

{current_abstraction_detail}

### CRITICAL: Name the COMMUNICATION ACTION, Not the Topic

The messages show WHAT someone is talking about, but you must name HOW they communicate.

**Example transformation:**
- Messages about "testing" → Name the behavior: "Provides Detailed Test Results" or "Asks for Test Coverage"
- Messages about "documentation" → Name the behavior: "Writes Step-by-Step Instructions" or "Requests Clarification"
- Messages about "bugs" → Name the behavior: "Describes Problems Thoroughly" or "Proposes Quick Fixes"

### Expected Pattern Types at This Level

{current_abstraction_examples}

### What Makes a GOOD Bottom-Level Pattern Name

**GOOD** (communication actions):
- "Asks Clarifying Questions" - observable action
- "Provides Step-by-Step Explanations" - how they communicate
- "Acknowledges Others' Input" - interaction behavior
- "Gives Specific Actionable Feedback" - communication style
- "Summarizes Decisions" - communication action
- "Requests Status Updates" - interaction pattern

**BAD** (topics, domains, or roles):
- "Testing Work" - topic, not behavior
- "MDM Documentation" - domain, not behavior
- "Twitter Scraper" - technical tool, not behavior
- "Marketing Handler" - role, not behavior
- "Database Manager" - job title, not behavior
- "Bug Fixer" - task type, not behavior

### The Key Question

For each pattern, ask: **"What COMMUNICATION ACTION does this represent?"**
- NOT "What technical domain is this about?"
- NOT "What project/tool is mentioned?"
- NOT "What job role does this sound like?"
"""

LEVEL_CONTEXT_MIDDLE = """
## Current Level: {level_name}

**Abstraction Level: {current_abstraction_degree}** - {current_abstraction_description}

You are naming patterns at a **middle level** of the hierarchy for **{encoder_name}**.

### What This Level Captures

{current_abstraction_detail}

### Composition from Previous Level

Each **{level_name}** pattern is composed of multiple **{prev_level}** patterns.
Look at the contributing {prev_level} patterns to understand what behavioral tendency they form together.

### CRITICAL: Name the BEHAVIORAL TENDENCY, Not the Domain

When multiple atomic actions combine, they reveal a behavioral tendency - a consistent WAY of working.

**Example transformation:**
- Multiple testing-related actions → "Thorough Verification Before Proceeding"
- Multiple documentation actions → "Proactive Knowledge Documentation"
- Multiple question-asking actions → "Seeks Understanding Before Acting"

### Expected Pattern Types at This Level

{current_abstraction_examples}

### What Makes a GOOD Mid-Level Pattern Name

**GOOD** (behavioral tendencies):
- "Thorough Analysis Before Responding" - how they approach problems
- "Proactive Status Communication" - communication tendency
- "Collaborative Problem-Solving" - working style
- "Constructive Feedback Delivery" - interaction tendency
- "Clear Escalation of Blockers" - communication pattern

**BAD** (domains or process names):
- "Testing and QA" - domain category
- "CI/CD Pipeline Work" - technical process
- "Code Review Management" - task category
- "Database Operations" - technical domain
- "Marketing Coordination" - functional area

### The Key Question

For each pattern, ask: **"What behavioral TENDENCY does this represent?"**
- NOT "What technical area do these activities fall under?"
- NOT "What process or workflow is this part of?"
"""

LEVEL_CONTEXT_FINAL_ENCODER = """
## Current Level: {level_name}

**Abstraction Level: {current_abstraction_degree}** - {current_abstraction_description}

You are naming patterns at the **final level** of **{encoder_name}**'s hierarchy.

### What This Level Captures

{current_abstraction_detail}

### Composition from Previous Level

Each **{level_name}** pattern is composed of multiple **{prev_level}** patterns.
Look at the contributing patterns to understand what higher-order behavioral characteristic they form.

### CRITICAL: Name the BEHAVIORAL CHARACTERISTIC, Not the Technical Specialty

At this level, patterns should represent how someone consistently operates across different situations.

**Example transformation:**
- Multiple "thorough" tendencies → "Consistently Detailed and Careful"
- Multiple "proactive" tendencies → "Anticipates Needs and Acts Early"
- Multiple "collaborative" tendencies → "Team-Centered Working Style"

### Expected Pattern Types at This Level

{current_abstraction_examples}

### What Makes a GOOD Top-Level Pattern Name

**GOOD** (behavioral characteristics):
- "Cross-Team Coordination Style" - how they work with others
- "Knowledge Sharing Orientation" - behavioral characteristic
- "Responsive Support to Teammates" - interaction style
- "Systematic Problem-Solving Approach" - working characteristic
- "Clear and Structured Communication" - communication characteristic

**BAD** (technical specialties or roles):
- "Backend System Expert" - technical specialty
- "DevOps Coordinator" - role/title
- "Data Pipeline Manager" - functional area
- "Testing Lead" - role, not behavior
- "Infrastructure Specialist" - technical domain

### The Key Question

For each pattern, ask: **"What BEHAVIORAL CHARACTERISTIC does this represent?"**
- NOT "What technical specialty is this?"
- NOT "What role or title does this sound like?"

**These patterns feed into the unified level**, which will combine them into broad evaluation dimensions.
"""

LEVEL_CONTEXT_UNIFIED = """
## Current Level: unified

**Abstraction Level: {current_abstraction_degree}** - {current_abstraction_description}

You are naming patterns at the **unified level** - these are the dimensions managers use to evaluate team members.

### What This Level Captures

{current_abstraction_detail}

### Critical Requirement: BROAD Evaluation Dimensions

These unified patterns are used to **score every engineer on the team**. They MUST be:

1. **Universal**: ANY engineer can be scored on this, regardless of specialty
   - A frontend dev, backend dev, DevOps engineer, or data scientist should ALL be scorable
   - "System Monitoring" is too specific - not everyone does this
   - "Communication Clarity" is universal - everyone communicates

2. **Evaluative**: Something a manager would put on a performance review
   - Ask: "Would a manager rate employees on this dimension?"
   - "Proactive Initiative" - YES, managers evaluate this
   - "Data Pipeline Optimization" - NO, too technical/specific

3. **Behavioral**: About HOW someone works, not WHAT they work on
   - "Thoroughness" - behavioral, applies to any task
   - "Database Management" - technical domain, not behavior

### Composition from All Encoders

Each unified pattern combines **{final_level}** patterns from all {num_encoders} encoders.
Look at the contributing patterns to understand what broad behavioral theme they represent together.

### Expected Pattern Types at This Level

{current_abstraction_examples}

### What Makes a GOOD Unified Pattern Name

**GOOD** (broad, evaluative, universal):
- "Communication Clarity" - everyone communicates, clarity is measurable
- "Collaborative Problem-Solving" - universal behavior
- "Proactive Initiative" - any engineer can show initiative
- "Responsiveness to Team Needs" - universal team behavior
- "Thoroughness in Work" - applies to any task
- "Knowledge Sharing" - any engineer can share knowledge

**BAD** (too specific, technical, or role-based):
- "System Monitoring Excellence" - not everyone monitors systems
- "Codebase Optimization" - too technical
- "Data-Driven Decision Making" - sounds like a specific role
- "Initiative Implementer" - role title, not behavior
- "Workflow Coordinator" - job function, not behavioral dimension

### The Test

For each pattern name, ask: **"Could I rate EVERY engineer on my team on this?"**
- If YES → good unified pattern
- If NO → too specific, try again
"""

# Map level types to their context templates
LEVEL_CONTEXTS = {
    "first": LEVEL_CONTEXT_FIRST,
    "middle": LEVEL_CONTEXT_MIDDLE,
    "final_encoder": LEVEL_CONTEXT_FINAL_ENCODER,
    "unified": LEVEL_CONTEXT_UNIFIED,
}

# =============================================================================
# MODEL STRUCTURE TEMPLATE - For dynamic generation
# =============================================================================

MODEL_STRUCTURE_HEADER = """
## Model Architecture

This VAE uses a **multi-encoder hierarchical architecture**:

- **{num_encoders} parallel encoders** process the same input independently
- Each encoder has **{num_levels} hierarchical levels**
- Lower levels capture fine-grained patterns
- Higher levels capture increasingly abstract patterns
- The **unified level** combines outputs from all encoders
"""

# =============================================================================
# COMPOSITION CONTEXT
# =============================================================================

COMPOSITION_HEADER = """
## Pattern Composition

Each pattern below shows which **{prev_level}** patterns contribute to it.
Use this information to understand what lower-level behaviors combine to form each pattern.
"""

# =============================================================================
# NAMING GUIDELINES
# =============================================================================

NAMING_GUIDELINES_TEMPLATE = """
## Naming Guidelines

### Name Requirements
- **Maximum {max_words} words** per name
- **Behavior-focused**: Focus on HOW the engineer operates, not WHAT they work on
- **Observable**: Something a manager could notice in interactions
- **Unique**: Each pattern name should be clearly distinct from others

### Good vs. Bad Examples

| Good Name | Bad Name | Why |
|-----------|----------|-----|
| "Detailed Code Reviews" | "Testing Work" | Behavior vs. topic/domain |
| "Proactive Status Updates" | "Bug Fixer" | How they communicate vs. task type |
| "Asks Clarifying Questions" | "Curious" | Observable action vs. personality trait |
| "Thorough Problem Analysis" | "Backend Developer" | Behavior vs. role/specialization |
| "Constructive Feedback Style" | "CI/CD Pipeline" | Communication pattern vs. technology |
| "Responsive to Team Requests" | "Good Communicator" | Specific behavior vs. vague praise |

### Common Mistakes to Avoid

**Topic/Domain Names** (BAD):
- "Testing Patterns", "Database Work", "API Development", "Frontend Features"
- These describe WHAT someone works on, not HOW they behave

**Task Type Names** (BAD):
- "Bug Fixes", "Code Reviews", "Documentation", "Deployments"
- These are task categories, not behavioral patterns

**Role Names** (BAD):
- "Tech Lead Behavior", "Junior Developer", "DevOps Engineer"
- Patterns should describe behavior, not job titles

**Personality Labels** (BAD):
- "Natural Leader", "Detail-Oriented", "Team Player"
- Too abstract - describe the specific behaviors instead

### Description Requirements
- 1-2 sentences explaining what BEHAVIOR the pattern captures
- Describe how someone with high scores on this pattern would act
- Focus on the communication/collaboration style, not the technical content
"""

# =============================================================================
# OUTPUT FORMAT
# =============================================================================

OUTPUT_FORMAT_HEADER = """
## Output Format

Return **ONLY** valid JSON with no markdown formatting, code blocks, or extra text.

Expected structure:
```json
{
"""

OUTPUT_FORMAT_FOOTER = """
```

**Important:**
- Include ALL patterns listed above
- Use the exact pattern keys shown (e.g., "bottom_0", "mid_1")
- Ensure valid JSON syntax
"""
