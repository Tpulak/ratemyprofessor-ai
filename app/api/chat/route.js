import {NextResponse} from 'next/server'
import { Pinecone } from '@pinecone-database/pinecone'

import OpenAI from 'openai'

const systemPrompt = 
`
You are an AI assistant specializing in helping students find professors based on their specific needs and preferences. Your knowledge base consists of professor reviews and ratings, which you'll use to provide personalized recommendations.

For each user query, you will:

1. Analyze the user's request to understand their requirements, preferences, and any specific criteria they mention.

2. Use RAG (Retrieval-Augmented Generation) to search your knowledge base and retrieve the most relevant professor information based on the query.

3. Select and present the top 3 professors that best match the user's criteria.

4. For each recommended professor, provide:
   - Name
   - Subject/Department
   - Overall rating (out of 5 stars)
   - A brief summary of their strengths and any potential drawbacks
   - A short excerpt from a relevant student review

5. After presenting the top 3 options, offer to provide more details on any of the recommended professors or to refine the search if needed.

Remember to maintain a helpful and informative tone, and always prioritize the student's needs and preferences in your recommendations. If a query is too vague or broad, ask follow-up questions to better understand the student's requirements.

Do not invent or fabricate information about professors. If you don't have enough information to confidently answer a query, inform the user and suggest how they might refine their search.

Your goal is to help students make informed decisions about their education by providing accurate, relevant, and helpful information about professors.
`

export async function POST(req){
    const data = await req.json()
    const pc = new Pinecone({
        apiKey: process.env.PINECONE_API_KEY,
    })
    const index = pc.index('rag').namespace('ns1')
    const openai = new OpenAI()

    const text = data[data.length - 1].content
    const embedding = await openai.Embeddings.create({
        model: 'text-embedding-3-small',
        input: text,
        encoding_format: 'float',
    })

    const results = await index.query({
        topK: 3,
        includeMetadata: true,
        vector: embedding.data[0].embedding
    })

    let resultString = 
        '\n\nReturned results from vector db (done automatically): '
    results.matches.forEach((match) => {
        resultString += `\n
        Professor: ${match.id}
        Review: ${match.metadata.stars}
        Subject: ${match.metadata.subject}
        Stars ${match.metadata.stars}
        \n\n
        `
    })

    const lastMessage = data[data.length - 1]
    const lastMessageContent = lastMessage.content + resultString
    const lastDataWithoutLastMessage = data.slice(0, data.length - 1)
    const completion = await openai.chat.completions.create({
        messages: [
            {role: 'system', content: systemPrompt},
            ...lastDataWithoutLastMessage,
            {role: 'user', content: lastMessageContent}
        ],
        model: 'gpt-4o-mini',
        stream: true,
    })

    const stream =  new ReadableStream({
        async start(controller){
            const encoder = new TextEncoder()
            try{
                for await (const chunk of completion){
                    const content = chunk.choices[0]?.delta?.content
                    if (content){
                        const text = encoder.encode(content)
                        controller.enqueue(text)
                    }
                }
            }
            catch(err){
                controller.error(err)
            }
            finally {
                controller.close()
            }
        },
    })

    return new NextResponse(stream)
}