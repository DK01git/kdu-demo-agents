import warnings
warnings.filterwarnings('ignore')
from crewai import Agent, Task, Crew
import os
from crewai_tools import SerperDevTool, ScrapeWebsiteTool,  WebsiteSearchTool
from IPython.display import Markdown
import gradio as gr
from openai import AzureOpenAI
from utils_azure import get_azure_credentials
from concurrent.futures import ThreadPoolExecutor
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential


# Initialize tools
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

# Get Azure credentials
azure_key, azure_endpoint = get_azure_credentials()
client = AzureOpenAI(
    api_key=azure_key,
    api_version="2024-02-15-preview",  # chnage the date if possible
    azure_endpoint=azure_endpoint
)

def generate_blog_post(topic, progress=gr.Progress()):
    total_steps = 6
    current_step = 0
    
    def update_progress():
        nonlocal current_step
        current_step += 1
        progress(current_step/total_steps, desc=f"Step {current_step}/{total_steps}")
    
    # Optimized LLM config
    llm_config = {
        "client": client,
        "model": "DeepSeek-R1-ozfcc",
        "temperature": 0.7,
        "max_tokens": 500,
        "request_timeout": 30
    }

    # Initialize agents with optimized config
    researcher = Agent(
        role="Digital Marketing Researcher",
        goal="Gather accurate, relevant, and engaging information on {topic} to support the creation of a high-quality blog post.",
        backstory="You are an expert in digital marketing research, specializing in identifying trends, data, and insights that resonate with the target audience.",
        instructions=[
            "1. Identify the target audience and their preferences for the topic: {topic}.",
            "2. Conduct thorough research using credible sources such as industry reports, case studies, and reputable websites.",
            "3. Gather data, statistics, and trends that support the topic and provide value to the audience.",
            "4. Organize the research into clear sections (e.g., introduction, key points, examples, and conclusions).",
            "5. Ensure all information is accurate, up-to-date, and properly cited.",
            "6. Provide a summary of the research findings to the Content Writer for further development."],
        llm_config=llm_config,
        allow_delegation=False,
        verbose=True,
        tools=[search_tool, scrape_tool]
    )
    update_progress()

    writer = Agent(
        role="Content Writer",
        goal="Write a compelling, engaging, and well-structured blog post on {topic} based on the research provided.",
        backstory="You are a skilled content writer with expertise in crafting blog posts that resonate with the target audience.",
        instructions=[
            "1. Review the research provided by the Researcher and identify the key points to include in the blog post.",
            "2. Write a captivating headline and introduction to grab the reader's attention.",
            "3. Structure the blog post into sections (e.g., introduction, main body, conclusion) for easy readability.",
            "4. Use clear, concise, and engaging language to communicate the information effectively.",
            "5. Incorporate relevant examples, statistics, and quotes from the research to support your points.",
            "6. Ensure the tone and style align with the company's brand guidelines.",
            "7. Optimize the content for SEO by including relevant keywords and meta descriptions.",
            "8. Submit the draft to the Editor for review and refinement."
        ],
        llm_config=llm_config,
        allow_delegation=False,
        verbose=True,
        tools=[search_tool, scrape_tool]
    )
    update_progress()

    editor = Agent(
        role="Content Editor",
        goal="Review and refine the blog post to ensure it is polished, error-free, appropriate for the company's policy and ready for publication.",
        backstory="You are an experienced content editor with a keen eye for detail and a deep understanding of the company's brand voice.",
        instructions=[
            "1. Review the blog post draft for grammar, spelling, and punctuation errors.",
            "2. Ensure the content is clear, concise, and easy to understand.",
            "3. Verify that the tone and style align with the company's brand guidelines.",
            "4. Check that the blog post is well-structured and flows logically.",
            "5. Confirm that all facts, statistics, and quotes are accurate and properly cited.",
            "6. Provide constructive feedback to the Content Writer if revisions are needed.",
            "7. Finalize the blog post and prepare it for publication."
        ],
        allow_delegation=False,
        verbose=True,
        tools=[search_tool, scrape_tool]
    )
    update_progress()
    
    # Keep original tasks but add progress tracking
    plan = Task(
        description=(
            "1. Analyze the topic: {topic} and identify its relevance to the target audience.\n"
            "2. Research the latest trends, statistics, and news related to {topic}.\n"
            "3. Identify key players, influencers, or brands in the industry related to {topic}.\n"
            "4. Gather credible sources such as industry reports, case studies, and reputable websites.\n"
            "5. Highlight pain points, challenges, and opportunities related to {topic}.\n"
            "6. Organize the research into clear sections (e.g., trends, data, examples, and sources)."
        ),
        expected_output=(
            "A detailed research document containing:\n"
            "- Key trends and statistics related to {topic}.\n"
            "- Credible sources and references.\n"
            "- Insights into the target audience's pain points and interests.\n"
            "- A structured outline of the research findings."
        ),
        tools=[search_tool, scrape_tool],
        agent=researcher,
    )
    update_progress()

    audience_analysis = Task(
        description=(
            "1. Define the target audience for the blog post on {topic}.\n"
            "2. Identify the audience's demographics, interests, and preferences.\n"
            "3. Analyze the audience's pain points and how {topic} can address them.\n"
            "4. Determine the tone and style that will resonate with the audience.\n"
            "5. Provide recommendations for engaging the audience effectively."
        ),
        expected_output=(
            "An audience analysis report containing:\n"
            "- Demographics and psychographics of the target audience.\n"
            "- Pain points and interests related to {topic}.\n"
            "- Recommendations for tone, style, and engagement strategies."
        ),
        tools=[search_tool, scrape_tool],
        agent=researcher,
    )
    update_progress()

    write = Task(
        description=(
            "1. Use the research document and audience analysis to create a blog draft.\n"
            "2. Write a compelling headline and introduction to grab the reader's attention.\n"
            "3. Structure the blog post into sections (e.g., introduction, main body, conclusion).\n"
            "4. Incorporate key points, examples, and statistics from the research.\n"
            "5. Use a tone and style that aligns with the target audience and brand guidelines.\n"
            "6. Include a clear call-to-action (CTA) to engage the audience.\n"
            "7. Optimize the content for SEO by including relevant keywords and meta descriptions."
        ),
        expected_output=(
            "A complete blog draft containing:\n"
            "- A captivating headline and introduction.\n"
            "- Well-structured sections with clear key points.\n"
            "- Examples, statistics, and quotes from the research.\n"
            "- A strong call-to-action (CTA).\n"
            "- SEO-optimized content with keywords and meta descriptions."
        ),
        tools=[search_tool, scrape_tool],
        agent=writer,
    )
    update_progress()

    revise_blog = Task(
        description=(
            "1. Review the Editor's feedback on the blog draft.\n"
            "2. Make necessary revisions to improve clarity, tone, and structure.\n"
            "3. Ensure all facts, statistics, and quotes are accurate and properly cited.\n"
            "4. Double-check SEO optimization, including keywords and meta descriptions.\n"
            "5. Submit the revised draft to the Editor for final approval."
        ),
        expected_output=(
            "A revised blog draft that incorporates all feedback and is ready for final review."
        ),
        tools=[search_tool, scrape_tool],
        agent=writer,
    )
    update_progress()

    edit = Task(
        description=(
            "1. Review the blog draft for grammar, spelling, and punctuation errors.\n"
            "2. Ensure the content is clear, concise, and easy to understand.\n"
            "3. Verify that the tone and style align with the company's brand guidelines.\n"
            "4. Check that the blog post is well-structured and flows logically.\n"
            "5. Confirm that all facts, statistics, and quotes are accurate and properly cited.\n"
            "6. Provide constructive feedback to the Writer for revisions if needed."
        ),
        expected_output=(
            "A reviewed blog draft with:\n"
            "- Corrections for grammar, spelling, and punctuation.\n"
            "- Suggestions for improving clarity, tone, and structure.\n"
            "- Confirmation of factual accuracy and proper citations.\n"
            "- Feedback for the Writer if revisions are required."
        ),
        tools=[search_tool, scrape_tool],
        agent=editor,
    )
    update_progress()

    finalize_blog = Task(
        description=(
            "1. Review the revised blog draft from the Writer.\n"
            "2. Ensure all feedback has been incorporated and the content is error-free.\n"
            "3. Confirm that the blog post is optimized for SEO and aligns with brand guidelines.\n"
            "4. Prepare the blog post for publication by formatting it appropriately.\n"
            "5. Submit the finalized blog post for publishing."
        ),
        expected_output=(
            "A finalized blog post that is:\n"
            "- Error-free and polished.\n"
            "- SEO-optimized and aligned with brand guidelines.\n"
            "- Ready for publication."
        ),
        tools=[search_tool, scrape_tool],
        agent=editor,
    )
    update_progress()

    # Use parallel processing in crew
    crew = Crew(
        agents=[researcher, writer, editor],
        tasks=[plan, audience_analysis, write, revise_blog, edit, finalize_blog],
        verbose=True,
        memory=True,
        process_parallel=True
    )

    result = crew.kickoff(inputs={"topic": topic})
    return result.raw


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def resilient_generate_blog(topic):
    try:
        return generate_blog_post(topic)
    except Exception as e:
        print(f"Retrying due to error: {str(e)}")
        raise

iface = gr.Interface(
    fn=resilient_generate_blog,
    inputs = gr.Textbox(lines=2, placeholder="Enter a topic for the blog post..."),
    outputs = gr.Textbox(lines=40, placeholder="Generated blog post will appear here..."),
    title = "AI Blog Post Generator",
    description = "Enter a topic to generate a SEO approved blog post."
)

iface.launch()