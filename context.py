from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import StreamingStdOutCallbackHandler

def create_chain():
    model = ChatOllama(
        model="llama3.2",
        temperature=0.3,  # Lower temperature for more controlled output
        callbacks=[StreamingStdOutCallbackHandler()],
        streaming=True
    )

    template = """
    STRICT STYLE TRANSFER RULES:
    1. OUTPUT LENGTH: Must produce EXACTLY the same number of sentences as the input context
    2. MEANING: Must convey EXACTLY the same information as the input context
    3. NO ADDITIONS: Do not add any new ideas, metaphors, or content
    4. STYLE ELEMENTS to copy from style text:
       - Word choice patterns
       - Sentence structure
       - Tone
    
    CONTEXT TEXT (THIS IS WHAT YOU'RE TRANSFORMING):
    >>> {context}
    Number of sentences in context: Count each sentence and match exactly.

    STYLE TEXT (ONLY FOR STYLE REFERENCE):
    >>> {style}

    BAD EXAMPLE:
    Original: "How are you?"
    Bad output: "How doth thy spirit fare on this dreary eve? The moon casts shadows upon my soul." (WRONG - added extra sentence)

    GOOD EXAMPLE:
    Original: "How are you?"
    Good output: "How doth thy spirit fare?" (CORRECT - same length, same meaning)

    YOUR TASK: Rewrite the context text in the style provided, maintaining EXACTLY:
    - The same number of sentences
    - The same information
    - The same basic meaning
    
    TRANSFORMED TEXT (MUST MATCH ORIGINAL LENGTH):"""

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {
            "context": lambda x: x["context"],
            "style": lambda x: x["style"]
        }
        | prompt
        | model
        | StrOutputParser()
    )
    
    return chain

def process_with_retry(chain, context_str, style_str, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = chain.invoke({
                "context": context_str,
                "style": style_str
            })
            return response
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed after {max_retries} attempts: {str(e)}")
                return None
            print(f"Attempt {attempt + 1} failed, retrying...")
            continue




def main():
    # Your existing context and style strings
    context_str = '''
    Thank you, dear Egeus. What’s going on with you?    
    '''  # (your full context string here)
    
    style_str = '''
        I have of late- but wherefore I know not- lost all my mirth, forgone all custom of exercises; and indeed, it goes so heavily with my disposition that this goodly frame, the earth, seems to me a sterile promontory; this most excellent canopy, the air, look you, this brave o'erhanging firmament, this majestical roof fretted with golden fire- why, it appeareth no other thing to me than a foul and pestilent congregation of vapours. What a piece of work is a man! how noble in reason! how infinite in faculties! in form and moving how express and admirable! in action how like an angel! in apprehension how like a god! the beauty of the world, the paragon of animals! And yet to me what is this quintessence of dust? Man delights not me- no, nor woman neither, though by your smiling you seem to say so...
        She should have died hereafter; There would’ve been a time for such a word. Tomorrow and tomorrow and tomorrow Creeps in this petty pace from day to day, To the last syllable of recorded time; And all our yesterday’s have lighted fools the way to dusty death. Out, out Brief Candle! Life’s but a walking shadow, a poor player That frets his hour upon the stage And then Is heard no more. It is a tale Told by an idiot, full of sound and fury Signifying nothing.
        O, that this too too solid flesh would melt

        Thaw and resolve itself into a dew!

        Or that the Everlasting had not fix’d

        His canon ‘gainst self-slaughter! O God! God!

        How weary, stale, flat and unprofitable,

        Seem to me all the uses of this world!

        Fie on’t! ah fie! ’tis an unweeded garden,

        That grows to seed; things rank and gross in nature

        Possess it merely. That it should come to this!

        ... and ...

        What a piece of work is a man!

        How noble in reason, how infinite in faculty

        In form and moving, how express and admirable

        In action, how like an angel

        In apprehension, how like a god!

        The beauty of the world

        The paragon of animals

        Duke

        Be absolute for death: either death or life

        Shall thereby be the sweeter. Reason thus with life:

        If I do lose thee, I do lose a thing

        That none but fools would keep. A breath thou art,

        Servile to all the skyey influences,

        That dost this habitation where thou keep’st

        Hourly afflict. Merely, thou art death’s fool,

        For him thou labor’st by thy flight to shun,

        And yet run’st toward him still. Thou art not noble,

        For all th’ accommodations that thou bear’st

        Are nurs’d by baseness. Thou’rt by no means valiant,

        For thou dost fear the soft and tender fork

        Of a poor worm. Thy best of rest is sleep,

        And that thou oft provok’st, yet grossly fear’st

        Thy death, which is no more. Thou art not thyself,

        For thou exists on many a thousand grains

        That issue out of dust. Happy thou art not,

        For what thou hast not, still thou striv’st to get,

        And what thou hast, forget’st. Thou art not certain,

        For thy complexion shifts to strange effects,

        After the moon. If thou art rich, thou’rt poor,

        For like an ass, whose back with ingots bows,

        Thou bear’st thy heavy riches but a journey,

        And death unloads thee. Friend hast thou none,

        For thine own bowels, which do call thee sire,

        The mere effusion of thy proper loins,

        Do curse the gout, serpigo, and the rheum

        For ending thee no sooner. Thou hast nor youth nor age,

        But as it were an after-dinner’s sleep,

        Dreaming on both, for all thy blessed youth

        Becomes as aged, and doth beg the alms

        Of palsied eld; and when thou art old and rich,

        Thou hast neither heat, affection, limb, nor beauty,

        To make thy riches pleasant. What’s yet in this

        That bears the name of life? Yet in this life

        Lie hid more thousand deaths; yet death we fear

        That makes these odds all even.
    '''  # (your full style string here)

    # Initialize chain
    chain = create_chain()
    if not chain:
        return
    
    # Process text
    result = process_with_retry(chain, context_str, style_str)
    if result:
        print("\nStyle Transfer Result:")
        print(result)
    else:
        print("Style transfer failed.")

if __name__ == "__main__":
    main()
