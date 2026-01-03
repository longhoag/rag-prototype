# RAG Pipeline Test Cases

This document contains test queries for The Alchemist RAG system with expected outputs.

## Test Query Format

Each test case includes:
- **Input**: The user's question
- **Expected Output**: Key information that should appear in the answer
- **Expected Sources**: Relevant chunk topics that should be retrieved

---

## Character Questions

### Test Case 1: Santiago's Identity
**Input:**
```
Who is Santiago?
```

**Expected Output:**
- Santiago is a shepherd boy
- He has recurring dreams about treasure
- He travels from Andalusia/Spain
- He seeks his Personal Legend

**Expected Sources:**
- Chunks describing Santiago as a shepherd
- His background and travels
- His character introduction

---

### Test Case 2: Fatima
**Input:**
```
Who is Fatima and what is her relationship with Santiago?
```

**Expected Output:**
- Fatima is a woman of the desert
- She lives at the Al-Fayoum oasis
- Santiago falls in love with her
- She encourages him to pursue his Personal Legend
- She waits for him while he continues his journey

**Expected Sources:**
- Santiago meeting Fatima at the well
- Their conversations about love and Personal Legend
- Fatima's role as desert woman

---

### Test Case 3: The Alchemist
**Input:**
```
Who is the alchemist and what does he teach Santiago?
```

**Expected Output:**
- The alchemist is a wise man who lives in the desert
- He knows the secrets of alchemy and the Soul of the World
- He teaches Santiago about listening to his heart
- He helps Santiago understand omens and the Language of the World
- He guides Santiago toward the Pyramids

**Expected Sources:**
- Santiago's meetings with the alchemist
- The alchemist's teachings
- Journey through the desert together

---

## Concept & Theme Questions

### Test Case 4: Personal Legend
**Input:**
```
What is a Personal Legend?
```

**Expected Output:**
- It is God's blessing or one's destiny
- It's what you have always wanted to accomplish
- Everyone knows their Personal Legend when they are young
- The universe conspires to help you achieve it
- It's a person's only real obligation

**Expected Sources:**
- The old king (Melchizedek) explaining Personal Legend
- Various characters discussing their dreams
- Philosophy about pursuing one's destiny

---

### Test Case 5: Soul of the World
**Input:**
```
What is the Soul of the World?
```

**Expected Output:**
- It's a universal force that connects all things
- Everything has a soul (minerals, vegetables, animals)
- It's the language that allows understanding of all creation
- When you want something, the Soul of the World conspires to help you
- It's accessed through understanding omens and the universal language

**Expected Sources:**
- The alchemist's teachings
- Discussions about interconnectedness
- Philosophy of alchemy

---

### Test Case 6: Omens
**Input:**
```
What are omens and why are they important?
```

**Expected Output:**
- Omens are signs from the universe/God
- They guide people toward their Personal Legend
- They are written in the Language of the World
- Examples: butterflies, hawks, desert signs
- One must learn to read and follow them

**Expected Sources:**
- Melchizedek teaching about omens
- Santiago learning to read signs
- Various omen examples throughout the story

---

## Plot & Journey Questions

### Test Case 7: Santiago's Dream
**Input:**
```
What is Santiago's recurring dream about?
```

**Expected Output:**
- He dreams of treasure near the Egyptian Pyramids
- A child appears and plays with his sheep
- The child transports him to the Pyramids
- The child tells him about hidden treasure
- He has this dream multiple times before acting on it

**Expected Sources:**
- Beginning of the story with dream description
- Santiago telling others about his dream
- Gypsy woman interpreting the dream

---

### Test Case 8: The Treasure Location
**Input:**
```
Where does Santiago ultimately find his treasure?
```

**Expected Output:**
- The treasure is back in Spain where he started
- It's buried under the sycamore tree in the abandoned church
- He has to travel to Egypt first to learn this
- A thief tells him about dreaming of treasure in Spain
- The journey was necessary for his growth

**Expected Sources:**
- The ending of the story
- The thief's revelation
- Santiago returning to the church

---

### Test Case 9: Crystal Merchant
**Input:**
```
What does Santiago learn from working at the crystal shop?
```

**Expected Output:**
- He learns about the Language of the World
- He discovers the principle of favorability (beginner's luck)
- He helps improve the merchant's business
- He learns that the merchant is afraid to pursue his own dream (Mecca)
- He saves money to continue his journey

**Expected Sources:**
- Santiago working at the crystal shop
- Conversations with the crystal merchant
- Business improvements Santiago implements

---

### Test Case 10: The Old King
**Input:**
```
Who is Melchizedek and what does he give to Santiago?
```

**Expected Output:**
- Melchizedek is the king of Salem
- He teaches Santiago about Personal Legend
- He gives Santiago Urim and Thummim (two stones for divination)
- He takes one-tenth of Santiago's flock
- He encourages Santiago to follow his dream

**Expected Sources:**
- Meeting with the old king in Tarifa
- The king's teachings
- Gift of the stones

---

## Symbolic & Philosophical Questions

### Test Case 11: Alchemy Symbolism
**Input:**
```
What does alchemy symbolize in the story?
```

**Expected Output:**
- Alchemy represents personal transformation
- The Master Work is about spiritual perfection
- Turning lead into gold symbolizes evolving to one's best self
- The journey itself is more important than the destination
- It's about understanding the Soul of the World

**Expected Sources:**
- The alchemist's teachings about alchemy
- Discussions of the Philosopher's Stone and Elixir of Life
- The Emerald Tablet

---

### Test Case 12: Love and Personal Legend
**Input:**
```
How does the book describe the relationship between love and pursuing one's Personal Legend?
```

**Expected Output:**
- True love never prevents someone from pursuing their Personal Legend
- Love is a stimulus, not an obstacle
- Fatima encourages Santiago to continue his journey
- Those who genuinely love you want you to be happy
- Love speaks the Language of the World

**Expected Sources:**
- Santiago and Fatima's relationship
- The alchemist's advice about love
- Conversations about pursuing dreams despite love

---

### Test Case 13: Fear and Dreams
**Input:**
```
What does the book say about fear and pursuing dreams?
```

**Expected Output:**
- Fear of failure prevents people from pursuing their Personal Legend
- Fear of suffering is worse than the suffering itself
- Most people give up when tested by the Soul of the World
- The darkest hour comes just before dawn
- People fear realizing their dreams after fighting for them

**Expected Sources:**
- The alchemist's teachings about fear
- Santiago's internal struggles
- The old king's warnings

---

## Setting & Location Questions

### Test Case 14: Al-Fayoum Oasis
**Input:**
```
What is the Al-Fayoum oasis and what happens there?
```

**Expected Output:**
- It's a large oasis in Egypt with date palms and wells
- Santiago meets Fatima there
- The caravan stops due to tribal wars
- Santiago meets the alchemist there
- It's considered neutral territory during war

**Expected Sources:**
- Arrival at the oasis
- Life at the oasis
- Meeting important characters there

---

### Test Case 15: The Journey Route
**Input:**
```
What is Santiago's journey route in the story?
```

**Expected Output:**
- Starts in Andalusia, Spain as a shepherd
- Travels to Tarifa and crosses to Tangier, Morocco
- Works in Tangier at the crystal shop
- Joins a caravan crossing the Sahara desert
- Reaches Al-Fayoum oasis
- Continues to the Pyramids in Egypt
- Returns to Spain at the end

**Expected Sources:**
- Various locations mentioned throughout
- Travel descriptions
- Geographic references

---

## Object & Symbol Questions

### Test Case 16: Urim and Thummim
**Input:**
```
What are Urim and Thummim?
```

**Expected Output:**
- Two stones given by the king of Salem
- One black (means "no"), one white (means "yes")
- Used for divination/reading omens
- From the king's golden breastplate
- Santiago rarely uses them, preferring to read omens himself

**Expected Sources:**
- The king giving the stones
- Biblical/historical references
- Santiago's use or non-use of the stones

---

### Test Case 17: The Emerald Tablet
**Input:**
```
What is the Emerald Tablet?
```

**Expected Output:**
- Ancient alchemical text
- Contains the essence of alchemy in a few lines
- Describes the Master Work
- States that everything is the manifestation of one thing
- Represents the wisdom passed down through generations

**Expected Sources:**
- The alchemist's teachings
- References to ancient wisdom
- Alchemy discussions

---

## Meta & Thematic Questions

### Test Case 18: Main Theme
**Input:**
```
What is the main message or theme of The Alchemist?
```

**Expected Output:**
- Follow your dreams and Personal Legend
- The journey is as important as the destination
- When you want something, the universe conspires to help you
- Listen to your heart and read the omens
- Treasure may be closer than you think
- Personal transformation through pursuing your destiny

**Expected Sources:**
- Multiple philosophical passages
- The alchemist's teachings
- The story's conclusion

---

### Test Case 19: Maktub
**Input:**
```
What does "Maktub" mean in the story?
```

**Expected Output:**
- Arabic word meaning "it is written"
- Represents the idea of destiny or fate
- Used to express acceptance of God's will
- Suggests everything happens for a reason
- Reflects the belief in predetermined destiny

**Expected Sources:**
- Arab characters using the term
- Crystal merchant's explanations
- References to fate and destiny

---

### Test Case 20: Language of the World
**Input:**
```
What is the Language of the World?
```

**Expected Output:**
- Universal language understood by all creation
- Language without words
- Includes omens, signs, and intuition
- Used by the desert, wind, sun, and all things
- Language of the heart and soul
- Connects everything in the universe

**Expected Sources:**
- The alchemist's teachings
- Santiago learning to communicate with nature
- Philosophy about universal understanding

---

## Usage Instructions

### Running Test Cases

```bash
# Single query test
poetry run python scripts/run_query.py "Who is Santiago?"

# Verbose mode for detailed output
poetry run python scripts/run_query.py -v "What is a Personal Legend?"

# Custom retrieval parameters
poetry run python scripts/run_query.py --top-k 5 --min-score 0.8 "What are omens?"

# Interactive mode for multiple tests
poetry run python scripts/run_query.py
```

### Evaluation Criteria

When testing, verify:
1. ✅ **Accuracy**: Answer contains key facts from expected output
2. ✅ **Relevance**: Retrieved chunks are topically related
3. ✅ **Completeness**: Major points are covered
4. ✅ **Source Citations**: Answer includes [Source N] references
5. ✅ **Coherence**: Response is well-structured and readable
6. ✅ **Similarity Scores**: Retrieved chunks have scores above threshold (default 0.7)

### Expected Performance

- **Retrieval**: Should return 10 chunks (default) with similarity > 0.7
- **Token Usage**: Typical query uses 150-400 prompt tokens, 50-200 completion tokens
- **Response Time**: Complete pipeline ~3-5 seconds (embedding + retrieval + generation)
- **Source Attribution**: Answers should cite 2-5 sources on average

---

## Notes

- These test cases are based on Paulo Coelho's "The Alchemist"
- The system has 78 indexed chunks from the complete text
- Answers may vary slightly based on model temperature and retrieved context
- Some questions may return overlapping chunks (e.g., character and theme questions)
- Adjust `--min-score` lower (e.g., 0.6) if retrieval returns too few results
- Use verbose mode (`-v`) to see which chunks were retrieved for debugging
