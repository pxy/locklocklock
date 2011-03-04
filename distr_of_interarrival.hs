
import Data.Ord
import Data.List
import System.IO
import Text.ParserCombinators.Parsec
import qualified Text.ParserCombinators.Parsec.Token as PT
import Text.ParserCombinators.Parsec.Language (emptyDef)


main2 =do dat <- getContents
          putStr . unwords . intersperse "\n" $ map show $ interArrival ((map read $ words dat) :: [Double])

interArrival :: (Num a) => [a] -> [a]
interArrival l = zipWith (-) (tail l) l

waittime :: (Num a) => [a] -> a
waittime [] = 0
waittime (x1:x2:xs) = x2 - x1 + waittime xs

main = do dat <- getContents
          parsed <- myparse dat
          putStrLn $ show $ (waittime (map fst (sortBy (comparing snd) parsed)) / ((fromIntegral (length parsed)) / 2))
          return ()

 -- parse it
myparse input = case parse file "" input of
	          Left er  -> return []
	          Right cl -> return $ cl
	 
-- | function generating a token parser based on a
-- lexical parsers combined with a language record definition
lexer :: PT.TokenParser st
lexer  = PT.makeTokenParser emptyDef                                  
 
integer :: CharParser st Integer
integer = PT.integer lexer                                            
 
double :: CharParser st Double
double = PT.float lexer

row :: CharParser st (Double, Integer)
row = do (char 'r' <|> char 't') >> space
         x1 <- double
         x2 <- integer
         return (x1, x2)
         
file :: CharParser st [(Double, Integer)]
file = many row


{-
6: 1660

7: 2080

total: 2400
-}