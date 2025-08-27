from langchain_core.runnables import RunnableLambda, RunnableParallel

runnable = RunnableLambda(lambda x: str(x))
# runnable.invoke(5)
runnable.batch([7, 8, 9])

def func(x):
    for y in x:
        yield str(y)

runnable = RunnableLambda(func)

for chunk in runnable.stream(range(5)):
    print(chunk)

runnable1 = RunnableLambda(lambda x: {"foo": x})
runnable2 = RunnableLambda(lambda x: [x] * 2)

chain = runnable1 | runnable2

print(chain.invoke(2)    )

runnable1 = RunnableLambda(lambda x: {"foo": x})
runnable2 = RunnableLambda(lambda x: [x] * 2)
runnable3 = RunnableLambda(lambda x: str(x))

chain = runnable1 | RunnableParallel(second=runnable2, third=runnable3)

chain.get_graph().print_ascii()